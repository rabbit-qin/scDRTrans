import numpy as np
import torch
from sklearn.metrics import *
import torch.nn.functional as F

def adentropy(F1, feat, lamda, eta=1.0):
    out_t1 = F1(feat, reverse=True, eta=eta)
    out_t1 = F.softmax(out_t1)
    loss_adent = lamda * torch.mean(torch.sum(out_t1 * (torch.log(out_t1 + 1e-5)), 1))
    return loss_adent

def save_AUCs(AUCs, filename):
    with open(filename, 'a') as f:
        f.write('\t'.join(map(str, AUCs)) + '\n')

def roc_auc_score_trainval(y_true, y_predicted):
    if len(np.unique(y_true)) == 1:
        return accuracy_score(y_true, np.rint(y_predicted))
    return roc_auc_score(y_true, y_predicted)


def cell_dim(drug, gene):
    global bulk_tasks, sc_tasks

    if drug == 'Erlotinib_GSE149214':
        if gene == "":
            bulk_tasks = {'expression': 3742, 'pathway': 128}
            sc_tasks = {'expression': 3742, 'pathway': 128}
        if gene == "HVGs":
            bulk_tasks = {'expression': 3742, 'pathway': 128}
            sc_tasks = {'expression': 3742, 'pathway': 128}

    if drug == 'Erlotinib_GSE149214_unknownDrug':
        if gene == "":
            bulk_tasks = {'expression': 3709, 'pathway': 128}
            sc_tasks = {'expression': 3709, 'pathway': 128}
        if gene == "HVGs":
            bulk_tasks = {'expression': 3709, 'pathway': 128}
            sc_tasks = {'expression': 3709, 'pathway': 128}

    if drug == 'Palbociclib':
        if gene == "":
            bulk_tasks = {'expression': 4024, 'pathway': 128}
            sc_tasks = {'expression': 4024, 'pathway': 128}
        if gene == "HVGs":
            bulk_tasks = {'expression': 4024, 'pathway': 128}
            sc_tasks = {'expression': 4024, 'pathway': 128}

    if drug == 'Palbociclib_GSE131984_Pal_unknownDrug':
        if gene == "":
            bulk_tasks = {'expression': 4029, 'pathway': 128}
            sc_tasks = {'expression': 4029, 'pathway': 128}
        if gene == "HVGs":
            bulk_tasks = {'expression': 4029, 'pathway': 128}
            sc_tasks = {'expression': 4029, 'pathway': 128}

    if drug == 'Etoposide':
        if gene == "":
            bulk_tasks = {'expression': 5782, 'pathway': 128}
            sc_tasks = {'expression': 5782, 'pathway': 128}
        if gene == "HVGs":
            bulk_tasks = {'expression': 5782, 'pathway': 128}
            sc_tasks = {'expression': 5782, 'pathway': 128}

    if drug == 'Etoposide_GSE149215_unknownDrug':
        if gene == "":
            bulk_tasks = {'expression': 5711, 'pathway': 128}
            sc_tasks = {'expression': 5711, 'pathway': 128}
        if gene == "HVGs":
            bulk_tasks = {'expression': 5711, 'pathway': 128}
            sc_tasks = {'expression': 5711, 'pathway': 128}

    if drug == 'Ibrutinib_GSE152469':
        if gene == "":
            bulk_tasks = {'expression': 2296, 'pathway': 128}
            sc_tasks = {'expression': 2296, 'pathway': 128}
        if gene == "HVGs":
            bulk_tasks = {'expression': 2296, 'pathway': 128}
            sc_tasks = {'expression': 2296, 'pathway': 128}

    if drug == 'Erlotinib_GSE134839':
        if gene == "":
            bulk_tasks = {'expression': 5591, 'pathway': 128}
            sc_tasks = {'expression': 5591, 'pathway': 128}
        if gene == "HVGs":
            bulk_tasks = {'expression': 5591, 'pathway': 128}
            sc_tasks = {'expression': 5591, 'pathway': 128}

    if drug == 'Erlotinib_GSE134839_unknownDrug':
        if gene == "":
            bulk_tasks = {'expression': 5570, 'pathway': 128}
            sc_tasks = {'expression': 5570, 'pathway': 128}
        if gene == "HVGs":
            bulk_tasks = {'expression': 5570, 'pathway': 128}
            sc_tasks = {'expression': 5570, 'pathway': 128}

    if drug == 'Cisplatin_Cisp568':
        if gene == "":
            bulk_tasks = {'expression': 2178, 'pathway': 128}
            sc_tasks = {'expression': 2178, 'pathway': 128}
        if gene == "HVGs":
            bulk_tasks = {'expression': 2178, 'pathway': 128}
            sc_tasks = {'expression': 2178, 'pathway': 128}

    if drug == 'Paclitaxel_GSE163836':
        if gene == "":
            bulk_tasks = {'expression': 3322, 'pathway': 128}
            sc_tasks = {'expression': 3322, 'pathway': 128}
        if gene == "HVGs":
            bulk_tasks = {'expression': 3322, 'pathway': 128}
            sc_tasks = {'expression': 3322, 'pathway': 128}

    if drug == 'Paclitaxel_GSE131984_Pac':
        if gene == "":
            bulk_tasks = {'expression': 2884, 'pathway': 128}
            sc_tasks = {'expression': 2884, 'pathway': 128}
        if gene == "HVGs":
            bulk_tasks = {'expression': 2884, 'pathway': 128}
            sc_tasks = {'expression': 2884, 'pathway': 128}

    if drug == 'Paclitaxel_GSE131984_Pac_unknownDrug':
        if gene == "":
            bulk_tasks = {'expression': 2888, 'pathway': 128}
            sc_tasks = {'expression': 2888, 'pathway': 128}
        if gene == "HVGs":
            bulk_tasks = {'expression': 2888, 'pathway': 128}
            sc_tasks = {'expression': 2888, 'pathway': 128}

    if drug == 'Crizotinib_ts485to_c':
        if gene == "":
            bulk_tasks = {'expression': 7052, 'pathway': 128}
            sc_tasks = {'expression': 7052, 'pathway': 128}
        if gene == "HVGs":
            bulk_tasks = {'expression': 7052, 'pathway': 128}
            sc_tasks = {'expression': 7052, 'pathway': 128}

    if drug == 'Crizotinib_fa34o_c':
        if gene == "":
            bulk_tasks = {'expression': 5818, 'pathway': 128}
            sc_tasks = {'expression': 5818, 'pathway': 128}
        if gene == "HVGs":
            bulk_tasks = {'expression': 5818, 'pathway': 128}
            sc_tasks = {'expression': 5818, 'pathway': 128}

    return bulk_tasks,sc_tasks

def create_dataset(x,y,batch_size,shuffle,sampler=None,drop_last=False):
    x = x.T
    x = torch.FloatTensor(x.values)
    y = y['response']
    y = torch.LongTensor(y.values)
    dataset = torch.utils.data.TensorDataset(x, y)
    dataset = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle,sampler=sampler) # ,drop_last=drop_last
    return dataset
