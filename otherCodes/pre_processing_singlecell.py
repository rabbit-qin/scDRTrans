import pandas as pd
import scanpy as sc
import numpy as np
import os
from sklearn.model_selection import train_test_split

drug = "Crizotinib"
directory = "/data/xyqin/singlecell2Drug/Data/scDrugAct/cell2drugLabel_cell2cell_Counts/"+drug+"_all.csv"
drug_data = pd.read_csv(directory, sep=',', index_col=None)
drug_data[['dataset', 'cell_barcodes']] = drug_data['cell'].str.split('@', n=1, expand=True)
drug_data_1 = drug_data[drug_data.label == 1]
drug_data_0 = drug_data[drug_data.label == 0]
downsampling_data_1, _ = train_test_split(drug_data_1, train_size=0.10, random_state=42)
downsampling_data_0, _ = train_test_split(drug_data_0, train_size=0.10, random_state=42)
drug_data_new = pd.concat([downsampling_data_1, downsampling_data_0], axis=0, ignore_index=True)
drug_data_new = drug_data_new.sample(frac=1, random_state=42)
drug_data_new = drug_data_new.reset_index(drop=True)

drug_data_filter = drug_data_new.drop_duplicates(subset=['cell_barcodes'], keep='first')
len(drug_data_filter['cell_barcodes'].value_counts())
datasets = drug_data_filter['dataset'].drop_duplicates().to_list()

Genes_valid_path = "/data/xyqin/singlecell2Drug/Data/scDrugAct/Genes_valid_scFundation.csv"
Genes_valid = pd.read_csv(Genes_valid_path, sep=',')

df = pd.DataFrame()
for i,dataset in enumerate(datasets):
    drug_cell_info = drug_data_filter.loc[drug_data_filter['dataset'] == dataset]
    dataset_path = "/data/xyqin/singlecell2Drug/Data/scDrugAct/ValidCountData/"
    dataset_path = dataset_path + dataset + ".csv"
    expression = pd.read_csv(dataset_path, sep=',', index_col=0)
    expre_select = expression[expression.columns[expression.columns.isin(drug_cell_info['cell_barcodes'].unique().tolist())]]
    expre_select = expre_select.loc[expre_select.index[expre_select.index.isin(Genes_valid['Genes'].tolist())]]
    expre_select.columns = dataset+'@'+expre_select.columns

    if i == 0:
        df = expre_select
    else:
        union_index = df.index.union(expre_select.index)
        df = df.reindex(union_index, fill_value=0)
        expre_select = expre_select.reindex(union_index, fill_value=0)
        df = pd.merge(df, expre_select, how='left', left_index=True, right_index=True)

df_filter = df.loc[df.index[df.index.isin(Genes_valid['Genes'].tolist())]]

path = "/data/xyqin/singlecell2Drug/Data/scDrugAct/cell2drugLabel_cell2cell_Counts_expr/"
df_filter_path = path + drug + "_filter_expr.csv"
df_filter.to_csv(df_filter_path, sep=',', index=True)

matadata_path = "/data/xyqin/singlecell2Drug/Data/scDrugAct/cell2drugLabel_cell2cell_Counts/"
metadata_filter_path = matadata_path + drug + ".csv"
drug_data_new = drug_data_new.drop(['dataset', 'cell_barcodes'], axis=1)
drug_data_new.to_csv(metadata_filter_path, sep=',', index=False)
