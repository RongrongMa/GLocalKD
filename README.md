# GLocalKD
This is the code for WSDM2022 paper "Deep Graph-level Anomaly Detection by Glocal Knowledge Distillation".

## Data Preparation

Some of datasets are put in ./dataset folder. Due to the large file size limitation, some datasets are not uploaded in this project. You may download them from the urls listed in the paper.

## Train

For datasets except HSE, p53, MMP, PPAR-gamma and hERG, run the following code. For datasets with node attributes, feature chooses default, otherwise deg-num.

	python main.py --dataset [] --feature [default/deg-num]

For HSE, p53, MMP and PPAR-gamma, run the following code.

	python main_Tox.py --dataset []

For hERG, run the following code.

	python main_smiles.py
