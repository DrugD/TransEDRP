# TransEDRP: Dual Transformer Model with Edge Embedding for Drug Response Prediction:

Open-sourced implementation for ICML 2023 Submission
TransEDRP is a dual transformer structure with edge embedding for drug response prediction.

# Python Dependencies

Our proposed TransEDRP framework is implemented in Python 3.6 and major libraries include:
- [rdkit-pypi](https://www.rdkit.org/)    2021.9.4  
- torch                      1.10.1
- torch-cluster              1.5.9
- torch-geometric            2.0.3
- torch-scatter              2.0.9
- torch-sparse               0.6.12
- torch-spline-conv          1.2.1
- torchvision                0.11.2

#  To Run:

Once the Python dependencies and path of datasets are fulfilled, use this command to run:

1. Create data in pytorch format. Move the downloaded dataset to `../root/data/` and pre-process it with the following command:

    `python project/TransEDRP/dataProcess/process_xxx.py`

2. Run the following command for model training and inference (Training **TransEDRP** on **GDSCv2** dataset as an example):

    `python main.py --config ../root/config/TransE/Transformer_edge_concat_GDSCv2.yaml`


# Datasets

All datasets used in this paper are downloaded and the raw files are under `../root/data/` dir. The original dataset can be found here:

- [GDSCv1](https://www.cancerrxgene.org/downloads/anova?screening_set=GDSC1)
- [GDSCv2](https://www.cancerrxgene.org/downloads/anova?screening_set=GDSC2)
- [CTRPv1](https://ctd2-data.nci.nih.gov/Public/Broad/)
- [CTRPv2](https://ctd2-data.nci.nih.gov/Public/Broad/)
- [GCSI](https://pharmacodb.pmgenomics.ca/datasets/4)

To facilitate the experiment, we package the original files of the above dataset and share file storage address at  [aliyun drive](
https://www.aliyundrive.com/s/GWBXCnmcp3V
) and [google drive](https://drive.google.com/file/d/1yz2Rw51cPuasnROwoHwpUSjuxOTcZsry/view?usp=share_link).

# Baselines

As provided in our overall experiments, all baselines and their URLs are:

- [tCNNs](https://github.com/Lowpassfilter/tCNNS-Project) : https://github.com/Lowpassfilter/tCNNS-Project
- [DeepCDR](https://github.com/kimmo1019/DeepCDR) : https://github.com/kimmo1019/DeepCDR
- [GraphDRP](https://github.com/hauldhut/GraphDRP) : https://github.com/hauldhut/GraphDRP
- [GraphTransDRP](https://github.com/chuducthang77/GraTransDRP) : https://github.com/chuducthang77/GraTransDRP
- [DeepTTC](https://github.com/jianglikun/DeepTTC) : https://github.com/jianglikun/DeepTTC

