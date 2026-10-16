import pandas as pd
import numpy as np
import os
import re

Genes_valid_path = "/home/public_data/xyqin/scDrug/SSDA2Drug/Genes_valid_scFundation.csv"
Genes_valid = pd.read_csv(Genes_valid_path, sep=',')

directory = "/home/public_data/xyqin/scDrug/SSDA2Drug/SSDA4Drug-singlecell/Datasets/rawData/GSE108397/PLX-4720-1036.csv"
file = pd.read_csv(directory, sep=',', index_col=0)
file = file.T

file['GeneNames'] = file.index
file['GeneNames'] = [i.strip("'") for i in file['GeneNames']]
file['GeneNames'] = file['GeneNames'].str.split('.').str[0]
exclude_cols = ['GeneNames']
numeric_cols = file.columns.difference(exclude_cols)
file['expression'] = file[numeric_cols].sum(axis=1)
file_filtered = file.loc[file.groupby('GeneNames')['expression'].idxmax()]
file_filtered.index = file_filtered['GeneNames']
numeric_cols = file.columns.difference(['GeneNames','expression'])
file_filtered = file_filtered[numeric_cols]

file_filtered = file_filtered[file_filtered.index.isin(['response','logIC50']+Genes_valid['Genes'].tolist())]
file_drop_genes = file[~file.index.isin(file_filtered.index)].index.tolist()
missing_genes = Genes_valid[~Genes_valid['Genes'].isin(file_filtered.index)]['Genes'].tolist()
if missing_genes:
    new_rows = pd.DataFrame(0, index=missing_genes, columns=file_filtered.columns)
    file_filtered = pd.concat([file_filtered, new_rows])
file_filtered = file_filtered.reindex(['response','logIC50']+Genes_valid['Genes'].tolist())

genes2num = dict(zip(Genes_valid['Genes'], Genes_valid['Encoder']))
index = [genes2num[x] for x in file_filtered.index[2:len(file_filtered.index)]]
file_filtered.index = ['response','logIC50']+index
file_filtered = file_filtered.T

df_filter_path = "/home/public_data/xyqin/scDrug/SSDA2Drug/SSDA4Drug-singlecell/Datasets/rawData/GSE108397/"
if not os.path.exists(df_filter_path):
    os.makedirs(df_filter_path, exist_ok=True)
    print(f"目录已创建：{df_filter_path}")
else:
    print(f"目录已存在：{df_filter_path}")
filename = "PLX-4720-1036_align.csv"
df_filter_path = os.path.join(df_filter_path, filename)
file_filtered.to_csv(df_filter_path, index=True)