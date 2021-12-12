# GLocalKD
This is the code for WSDM2022 paper "Deep Graph-level Anomaly Detection by Glocal Knowledge Distillation".

## Brief Introduction
Global and Local Knowledge Distillation (GLocalKD) is introduced in our WSDM22 paper, which leverages a set of normal training data to perform end-to-end anomaly score learning for graph-level anomaly detection (GAD) problem. GAD describes the problem of detecting graphs that are abnormal in their structure and/or the features of their nodes, as compared to other graphs. GLocalKD addresses a semi-supervised GAD problem in that the data known are all labeled normal data. The experiment results show that  GLocalKD can be implemented data-effectively and is robustness to anomaly contamination, indicating its applicability in both unsupervised (anomaly-contaminated unlabeled training data) and semi-supervised (exclusively normal training data) settings.

## Data Preparation

Some of datasets are put in ./dataset folder. Due to the large file size limitation, some datasets are not uploaded in this project. You may download them from the urls listed in the paper.

## Train

For datasets except HSE, p53, MMP, PPAR-gamma and hERG, run the following code. For datasets with node attributes, feature chooses default, otherwise deg-num.

	python main.py --dataset [] --feature [default/deg-num]

For HSE, p53, MMP and PPAR-gamma, run the following code.

	python main_Tox.py --dataset []

For hERG, run the following code.

	python main_smiles.py


## Citation
```bibtex
@inproceedings{ma2022deep,
  title={Deep Graph-level Anomaly Detection by Glocal Knowledge Distillation},
  author={Ma, Rongrong and Pang, Guansong and Chen, Ling and van den Hengel, Anton},
  booktitle={WSDM '22: The Fifteenth ACM International Conference on Web Search and Data Mining},
  year={2022}
}
```
