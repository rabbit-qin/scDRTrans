import pandas as pd
import scanpy as sc
import numpy as np
import os
from sklearn.model_selection import train_test_split

drug = "Crizotinib"
directory = "/data/xyqin/singlecell2Drug/Data/scDrugAct/cell2drugLabel_cell2cell_Counts/"+drug+".csv"
drug_data = pd.read_csv(directory, sep=',', index_col=None)
drug_data[['dataset', 'cell_barcodes']] = drug_data['cell'].str.split('@', n=1, expand=True)
drug_data_filter = drug_data.drop_duplicates(subset=['cell_barcodes'], keep='first')
len(drug_data_filter['cell_barcodes'].value_counts())
datasets = drug_data_filter['dataset'].drop_duplicates().to_list()
remove_list = ["GSE171145", "NSCLC_GSE146100", "NSCLC_GSE149655", "NSCLC_GSE179373", "GSE202371_LC", "NSCLC_GSE127471", "NSCLC_GSE153935", "NSCLC_GSE162498", "NSCLC_GSE99254", "NSCLC_GSE127465"]
#OSCC：remove_list = ["GSE103322", "HNSC_GSE180268", "GSE164690", "GSE195832", "LSCC_GSE150321", "NPC_GSE150430"]
#BC：remove_list = ["TNBC-021", "TNBC-008", "DCIS-028", "GSE180286", "BRCA_GSE150660", "BRCA_GSE138536", "BRCA_GSE114727_inDrop", "BRCA_GSE114727_10X", "BRCA_GSE110686"]
#CLL：remove_list = ["CLL_GSE111014", "CLL_GSE132065", "CLL_GSE142744", "CLL_GSE152469", "CLL_GSE159417", "CLL_GSE125881"]
#LUAD: remove_list = ["GSE171145", "NSCLC_GSE146100", "NSCLC_GSE149655", "NSCLC_GSE179373", "GSE202371_LC"]
#NSCLC: remove_list = ["GSE171145", "NSCLC_GSE146100", "NSCLC_GSE149655", "NSCLC_GSE179373", "GSE202371_LC", "NSCLC_GSE127471", "NSCLC_GSE153935", "NSCLC_GSE162498", "NSCLC_GSE99254", "NSCLC_GSE127465"]
datasets = [x for x in datasets if x not in remove_list]

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

path = "/data/xyqin/singlecell2Drug/Data/scDrugAct/cell2drugLabel_cell2cell_Counts_expr_unknownDrug/"
df_filter_path = path + drug + "_filter_expr_unknownDrug.csv"
df_filter.to_csv(df_filter_path, sep=',', index=True)

drug_data_filter = drug_data_filter.reset_index(drop=True)
drug_data_metadata = drug_data_filter.iloc[drug_data_filter.index[drug_data_filter['cell'].isin(df_filter.columns)]]
matadata_path = "/data/xyqin/singlecell2Drug/Data/scDrugAct/cell2drugLabel_cell2cell_Counts_unknownDrug/"
metadata_filter_path = matadata_path + drug + ".csv"
drug_data_metadata.to_csv(metadata_filter_path, sep=',', index=False)
