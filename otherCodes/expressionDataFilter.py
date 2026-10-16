import pandas as pd
import numpy as np
import os
import re

Genes_valid_path = "./Genes_valid_scFundation.csv"
Genes_valid = pd.read_csv(Genes_valid_path, sep=',')

directory = "./Datasets/rawData/GSE108397/GSE108397_Counts.csv"
file = pd.read_csv(directory, sep=',', index_col=0)

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

file_filtered = file_filtered[file_filtered.index.isin(Genes_valid['Genes'])]
file_drop_genes = file[~file.index.isin(file_filtered.index)].index.tolist()
missing_genes = Genes_valid[~Genes_valid['Genes'].isin(file_filtered.index)]['Genes'].tolist()
if missing_genes:
    new_rows = pd.DataFrame(0, index=missing_genes, columns=file_filtered.columns)
    file_filtered = pd.concat([file_filtered, new_rows])
file_filtered = file_filtered.reindex(Genes_valid['Genes'])

genes2num = dict(zip(Genes_valid['Genes'], Genes_valid['Encoder']))
file_filtered.index = [genes2num[x] for x in file_filtered.index]

df_filter_path = "./Datasets/rawData/GSE108397/"
if not os.path.exists(df_filter_path):
    os.makedirs(df_filter_path, exist_ok=True)
    print(f"目录已创建：{df_filter_path}")
else:
    print(f"目录已存在：{df_filter_path}")
filename = "GSE108397_align.csv"
df_filter_path = os.path.join(df_filter_path, filename)
file_filtered.to_csv(df_filter_path, index=True)

binary_Values_path = "./Datasets/rawData/GSE108397/GSE108397_metadata.csv"
binary_Values = pd.read_csv(binary_Values_path, sep=',')
binary_Values = binary_Values[['cell','drug']]
binary_Values['drug'] = binary_Values['drug'].replace('sensitive',1)
binary_Values['drug'] = binary_Values['drug'].replace('resistant',0)
print(binary_Values['drug'].value_counts())
binary_Values = binary_Values[binary_Values['cell'].isin(file_filtered.columns)]
binary_Values_filter_path = "./Datasets/rawData/GSE108397/"
filename = "GSE108397_meta_align.csv"
binary_Values_path = os.path.join(binary_Values_filter_path, filename)
binary_Values.to_csv(binary_Values_path, sep=",", index=False)