import pandas as pd
import scanpy as sc
import numpy as np
import os

directory = "/data/xyqin/singlecell2Drug/Data/scDrugAct/cell2drugLabel_cell2cell_Counts_expr/Cisplatin_filter_expr.csv"
source_data = pd.read_csv(directory, sep=',', index_col=0)
source_data = source_data.T

metadata_path = "/data/xyqin/singlecell2Drug/Data/scDrugAct/cell2drugLabel_cell2cell_Counts/Cisplatin.csv"
source_meta_data = pd.read_csv(metadata_path, sep=',', index_col=0)
source_meta_data = source_meta_data.drop(["drug"], axis=1)
source_meta_data.columns = ["response"]

source_data.index = source_data.index.astype(str)
source_data.columns = source_data.columns.astype(str)
adata_source = sc.AnnData(source_data)

print(adata_source)
print(adata_source.obs.head())
print(adata_source.var.head())
print(adata_source.X.shape)

drug = "Cisplatin"
dataset = "GSE117872_Cisp568"
path = "./Datasets/rawData" + '/'  + dataset
target_data = pd.read_csv(path + '/' + dataset + '_Counts.csv', sep=',',index_col=0)
target_data = target_data.T
target_data['GeneNames'] = target_data.index
target_data['GeneNames'] = [i.strip("'") for i in target_data['GeneNames']]
target_data['GeneNames'] = target_data['GeneNames'].str.split('.').str[0]
exclude_cols = ['GeneNames']
numeric_cols = target_data.columns.difference(exclude_cols)
target_data['expression'] = target_data[numeric_cols].sum(axis=1)
file_filtered = target_data.loc[target_data.groupby('GeneNames')['expression'].idxmax()]
file_filtered.index = file_filtered['GeneNames']
numeric_cols = target_data.columns.difference(['GeneNames','expression'])
file_filtered = file_filtered[numeric_cols]
file_filtered = file_filtered.T
adata_target = sc.AnnData(file_filtered)

target_metadata = pd.read_csv(path + '/' + dataset + '_metadata.csv', sep=',',index_col=0)
print(target_metadata['response'].value_counts())

def pre_processing(adata,type):
    if type == 'singlecell_complement':
        adata.var['mt'] = adata.var_names.str.startswith('MT-')
        sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
        adata = adata[adata.obs.pct_counts_mt < 20, :]
        sc.pp.highly_variable_genes(adata)
        highly_variable_genes = adata.var[adata.var['highly_variable']].index.tolist()
        print(f"筛选出的高变基因数量：{len(highly_variable_genes)}")
    elif type == 'singlecell':
        sc.pp.filter_cells(adata, min_genes=200)
        sc.pp.filter_genes(adata, min_cells=3)
        adata.var['mt'] = adata.var_names.str.startswith('MT-')
        sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
        adata = adata[adata.obs.pct_counts_mt < 20, :]
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata)
        highly_variable_genes = adata.var[adata.var['highly_variable']].index.tolist()
        print(f"筛选出的高变基因数量：{len(highly_variable_genes)}")
    return adata

adata_source = pre_processing(adata_source,'singlecell')
expr_matrix_source = pd.DataFrame(adata_source.X, index=adata_source.obs.index, columns=adata_source.var.index)
highly_variable_genes_source = adata_source.var[adata_source.var['highly_variable']].index.tolist()
adata_target = pre_processing(adata_target,'singlecell_complement')
expr_matrix_target = pd.DataFrame(adata_target.X, index=adata_target.obs.index, columns=adata_target.var.index)
highly_variable_genes_target = adata_target.var[adata_target.var['highly_variable']].index.tolist()

directory = "/data/xyqin/singlecell2Drug/Data/scDrugAct/senstive_gene_list.txt"
senstive_genes = pd.read_csv(directory, sep='\t', index_col=None, header=None)

retained_genes = set(senstive_genes.iloc[:,0]) | set(highly_variable_genes_source) | set(highly_variable_genes_target)
expr_matrix_source_filter = expr_matrix_source[expr_matrix_source.columns[expr_matrix_source.columns.isin(list(retained_genes))]]
expr_matrix_target_filter = expr_matrix_target[expr_matrix_target.columns[expr_matrix_target.columns.isin(list(retained_genes))]]
intersect_genes = set(expr_matrix_source_filter.columns) & set(expr_matrix_target_filter.columns)
expr_matrix_source_filter = expr_matrix_source_filter[expr_matrix_source_filter.columns[expr_matrix_source_filter.columns.isin(list(intersect_genes))]]
expr_matrix_target_filter = expr_matrix_target_filter[expr_matrix_target_filter.columns[expr_matrix_target_filter.columns.isin(list(intersect_genes))]]

Genes_valid = pd.DataFrame({"Genes": list(intersect_genes), "Encoder": range(len(list(intersect_genes)))})
genes2num = dict(zip(Genes_valid['Genes'], Genes_valid['Encoder']))
expr_matrix_source_filter.columns = [genes2num[x] for x in expr_matrix_source_filter.columns]
expr_matrix_target_filter.columns = [genes2num[x] for x in expr_matrix_target_filter.columns]
Genes_valid.to_csv(path + '/Genes2Encoder.csv', sep=',', index=False)

source_meta_data.index = source_meta_data.index.astype(str)
expr_matrix_source_filter.index = expr_matrix_source_filter.index.astype(str)
source_meta_data = source_meta_data.reindex(expr_matrix_source_filter.index)
source_data = pd.concat([source_meta_data, expr_matrix_source_filter], axis=1)
source_data.to_csv(path + '/' + drug + '_align.csv', sep=',', index=True)

target_metadata = target_metadata[target_metadata.index.isin(expr_matrix_target_filter.index)]
print(target_metadata['response'].value_counts())
expr_matrix_target_filter.to_csv(path + '/' + dataset + '_align.csv', sep=',', index=True)
target_metadata.to_csv(path + '/' + dataset + '_meta_align.csv', sep=',', index=True)