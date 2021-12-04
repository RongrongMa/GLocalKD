# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 17:06:01 2021

@author: MaRongrong
"""

import numpy as np
from sklearn.utils.random import sample_without_replacement
from sklearn.metrics import auc, precision_recall_curve, roc_curve
from sklearn.svm import OneClassSVM
import argparse
import networkx as nx
from GCN_embedding import GcnEncoderGraph_teacher, GcnEncoderGraph_student
import torch
import torch.nn as nn
import time
import GCN_embedding
from torch.autograd import Variable
from graph_sampler import GraphSampler
from numpy.random import seed
import random
import matplotlib.pyplot as plt
import copy
import torch.nn.functional as F
from sklearn.manifold import TSNE
from matplotlib import cm
from tdc.utils import retrieve_label_name_list
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from tdc.single_pred import Tox
from pysmiles import read_smiles

def arg_parse():
    parser = argparse.ArgumentParser(description='GLocalKD Arguments.')
    parser.add_argument('--datadir', dest='datadir', default ='dataset', help='Directory where benchmark is located')
    parser.add_argument('--DS', dest='DS', default ='hERG', help='dataset name')
    parser.add_argument('--max-nodes', dest='max_nodes', type=int, default=0, help='Maximum number of nodes (ignore graghs with nodes exceeding the number.')
    parser.add_argument('--clip', dest='clip', default=0.1, type=float, help='Gradient clipping.')
    parser.add_argument('--num_epochs', dest='num_epochs', default=150, type=int, help='total epoch number')
    parser.add_argument('--batch-size', dest='batch_size', default=300, type=int, help='Batch size.')
    parser.add_argument('--hidden-dim', dest='hidden_dim', default=512, type=int, help='Hidden dimension')
    parser.add_argument('--output-dim', dest='output_dim', default=256, type=int, help='Output dimension')
    parser.add_argument('--num-gc-layers', dest='num_gc_layers', default=3, type=int, help='Number of graph convolution layers before each pooling')
    parser.add_argument('--nobn', dest='bn', action='store_const', const=False, default=True, help='Whether batch normalization is used')
    parser.add_argument('--dropout', dest='dropout', default=0.3, type=float, help='Dropout rate.')
    parser.add_argument('--nobias', dest='bias', action='store_const', const=False, default=True, help='Whether to add bias. Default to True.')
    parser.add_argument('--feature', dest='feature', default='deg-num', help='use what node feature')
    parser.add_argument('--seed', dest='seed', type=int, default=1, help='seed')
    return parser.parse_args()

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True


def train(dataset, data_test_loader, model_teacher, model_student, args):    
    optimizer = torch.optim.Adam(filter(lambda p : p.requires_grad, model_student.parameters()), lr=0.0001)
    scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=50, gamma=0.5)
    epochs=[]
    auroc_final = 0
    for epoch in range(args.num_epochs):
        total_time = 0
        total_loss = 0.0
        model_student.train()

        for batch_idx, data in enumerate(dataset):           
            begin_time = time.time()
            model_student.zero_grad()
            adj = Variable(data['adj'].float(), requires_grad=False).cuda()
            h0 = Variable(data['feats'].float(), requires_grad=False).cuda()
           
            embed_node, embed = model_student(h0, adj)
            embed_teacher_node, embed_teacher = model_teacher(h0, adj)
            embed_teacher =  embed_teacher.detach()
            embed_teacher_node = embed_teacher_node.detach()
            loss_node = torch.mean(F.mse_loss(embed_node, embed_teacher_node, reduction='none'), dim=2).mean(dim=1).mean(dim=0)
            loss = F.mse_loss(embed, embed_teacher, reduction='none').mean(dim=1).mean(dim=0)
            loss = loss + loss_node
            
            loss.backward(loss.clone().detach())
            nn.utils.clip_grad_norm_(model_student.parameters(), args.clip)
            optimizer.step()
            scheduler.step()
            total_loss += loss
            elapsed = time.time() - begin_time
            total_time += elapsed
        
        if (epoch+1)%10 == 0 and epoch > 0:
            epochs.append(epoch)
            model_student.eval()   
            loss = []
            y=[]
            emb=[]
            
            for batch_idx, data in enumerate(data_test_loader):
               adj = Variable(data['adj'].float(), requires_grad=False).cuda()
               h0 = Variable(data['feats'].float(), requires_grad=False).cuda()
                        
               embed_node, embed = model_student(h0, adj)
               embed_teacher_node, embed_teacher = model_teacher(h0, adj)
               loss_node = torch.mean(F.mse_loss(embed_node, embed_teacher_node, reduction='none'), dim=2).mean(dim=1)
               loss_graph = F.mse_loss(embed, embed_teacher, reduction='none').mean(dim=1)
               loss_ = loss_graph + loss_node
               loss_ = np.array(loss_.cpu().detach())
               loss.append(loss_)
               if data['label'] == 0:
                   y.append(1)
               else:
                   y.append(0)            
               emb.append(embed.cpu().detach().numpy())
            
            label_test = []
            for loss_ in loss:
               label_test.append(loss_)
            label_test = np.array(label_test)
                                 
            fpr_ab, tpr_ab, _ = roc_curve(y, label_test)
            test_roc_ab = auc(fpr_ab, tpr_ab)   
            print('semi-supervised abnormal detection: auroc_ab: {}'.format(test_roc_ab))
        if epoch == (args.num_epochs-1):           
            auroc_final = test_roc_ab
    return auroc_final

    
if __name__ == '__main__':
    args = arg_parse()
    DS = args.DS
    setup_seed(args.seed)

    data = Tox(name = DS)
    df = data.get_data()
    data_=df.values
    smilesstr=[]
    graph_label=[]
    for i in range(data_.shape[0]):
        smilesstr.append(data_[i,1])
        graph_label.append(data_[i,2])

    graphs=[]
    for data in smilesstr:
        graphs.append(read_smiles(data))
    max_nodes_num = max([G.number_of_nodes() for G in graphs])
    datanum = len(graphs)
    print(datanum)
    
    for idx, graph in enumerate(graphs):
        graph.graph['label'] = graph_label[idx]
        
    kfd=StratifiedKFold(n_splits=5, random_state=args.seed, shuffle = True)
    result_auc=[]
    for k, (train_index,test_index) in enumerate(kfd.split(graphs, graph_label)):
        graphs_train_ = [graphs[i] for i in train_index]
        graphs_test = [graphs[i] for i in test_index]
       
        graphs_train = []
        for graph in graphs_train_:
            if graph.graph['label'] == 1:
                graphs_train.append(graph)
        

        num_train = len(graphs_train)
        num_test = len(graphs_test)
        print(num_train, num_test)
        
        dataset_sampler_train = GraphSampler(graphs_train, features=args.feature, normalize=False, max_num_nodes=max_nodes_num)
    
        model_teacher = GCN_embedding.GcnEncoderGraph_teacher(dataset_sampler_train.feat_dim, args.hidden_dim, args.output_dim, 2,
                args.num_gc_layers, bn=args.bn, args=args).cuda()
        for param in model_teacher.parameters():
            param.requires_grad = False
   
        model_student = GCN_embedding.GcnEncoderGraph_student(dataset_sampler_train.feat_dim, args.hidden_dim, args.output_dim, 2,
                args.num_gc_layers, bn=args.bn, args=args).cuda()
        
        data_train_loader = torch.utils.data.DataLoader(dataset_sampler_train, 
                                                    shuffle=True,
                                                    batch_size=args.batch_size)

    
        dataset_sampler_test = GraphSampler(graphs_test, features=args.feature, normalize=False, max_num_nodes=max_nodes_num)
        data_test_loader = torch.utils.data.DataLoader(dataset_sampler_test, 
                                                        shuffle=False,
                                                        batch_size=1)
        result = train(data_train_loader, data_test_loader, model_teacher, model_student, args)     
        result_auc.append(result)
            
    result_auc = np.array(result_auc)    
    auc_avg = np.mean(result_auc)
    auc_std = np.std(result_auc)
    print('auroc{}, average: {}, std: {}'.format(result_auc, auc_avg, auc_std))
    
    
    