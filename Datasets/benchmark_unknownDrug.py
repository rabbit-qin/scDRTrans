import pandas as pd
import numpy as np
import os

drug = "Ibrutinib"
dataset = "GSE152469"
path = "./Datasets/rawData" + '/'  + dataset + "_unknownDrug"
source_data=pd.read_csv(path + '/' + drug + '_align.csv', sep=',',index_col=0)
source_meta_data = source_data[["response"]]
print(source_meta_data['response'].value_counts())

source_save_path = "./Datasets/processedData" + '/'  + drug + '_GSE152469_unknownDrug/' + "source_data"
if not os.path.exists(source_save_path):
    os.makedirs(source_save_path)

source_meta_data.to_csv(source_save_path + '/source_meta_data.csv')
source_data = source_data.drop(["response"],axis = 1)
source_data = source_data.T
source_data.to_csv(source_save_path + '/source_exp_data.csv')

target_data = pd.read_csv(path + '/' + dataset + '_align.csv', sep=',',index_col=0)
target_meta_data = pd.read_csv(path + '/' + dataset + '_meta_align.csv', sep=',',index_col=0)
target_meta_data.columns = ["response"]
print(target_meta_data['response'].value_counts())

target_save_path = "./Datasets/processedData" + '/'  + drug  + "_GSE152469_unknownDrug/target_data"
if not os.path.exists(target_save_path):
    os.makedirs(target_save_path)

target_meta_data.to_csv(target_save_path + '/target_meta_data.csv')
target_data = target_data.T
target_data.to_csv(target_save_path + '/target_exp_data.csv')

source_data = source_data.T
target_data = target_data.T
source_data.columns = source_data.columns.astype(int)
target_data.columns = target_data.columns.astype(int)
combine = pd.concat([source_data, target_data],axis=0)

from sklearn.preprocessing import StandardScaler
import pandas as pd

scaler = StandardScaler()
scaler.fit(combine)

source_scaled = pd.DataFrame(scaler.transform(source_data), columns=source_data.columns)
target_scaled = pd.DataFrame(scaler.transform(target_data), columns=target_data.columns)
source_scaled.index = source_data.index
source_scaled = source_scaled.T
source_scaled.to_csv(source_save_path + '/source_scaled.csv')
target_scaled.index = target_data.index
target_scaled = target_scaled.T
target_scaled.to_csv(target_save_path + '/target_scaled.csv')