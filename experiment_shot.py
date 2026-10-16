import argparse
import warnings
import os
import numpy as np
import pandas as pd
import torch
from sympy.physics.vector.tests.test_printing import alpha
import utils
import trainer
import model as m
import matplotlib
import random
import itertools
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
random.seed(seed)
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
# slower and more reproducible
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def main(args):
    writer = SummaryWriter(args.log_dir)
    x_source = pd.read_csv("./Datasets/processedData" + '/' + args.drug + "/source_data/source_scaled.csv", index_col=0)
    y_source = pd.read_csv("./Datasets/processedData" + '/' + args.drug + "/source_data/source_meta_data.csv", index_col=0)
    x_source = x_source.T
    for i in range(50):
        epochs = args.epochs
        lr = args.lr
        dropout = args.dropout
        batch_size = args.batch_size

        preditor_hdims = args.predictor_h_dims.split(",")
        preditor_hdims = list(map(int, preditor_hdims))
        encoder_hdims = args.encoder_h_dims.split(",")
        encoder_hdims = list(map(int, encoder_hdims))

        para = ("drug_" + str(args.drug) + "_method_" + str(args.method) + "_gene_" + str(args.gene) + "_DAEdim_" + str(args.encoder_h_dims) + "_Predim_" +
                str(args.predictor_h_dims) + "_dropout_" + str(args.dropout) + "_lr_" + str(args.lr) + "_bs_" + str(args.batch_size))
        print("para:", para)

        x_train_source, x_val_source, y_train_source, y_val_source = train_test_split(x_source, y_source, test_size=0.2, random_state=i)
        x_train_source = x_train_source.T
        x_val_source = x_val_source.T

        from collections import Counter
        class_sample_count = np.array([Counter(y_source['response'])[0] / len(y_source['response']), Counter(y_source['response'])[1] / len(y_source['response'])])
        weight = 1. / class_sample_count
        samples_weight = np.array([weight[t] for t in y_train_source["response"].values])
        samples_weight = torch.from_numpy(samples_weight)
        samples_weight = samples_weight.reshape(-1)
        from torch.utils.data.sampler import WeightedRandomSampler
        sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight), replacement=True)
        source_train = utils.create_dataset(x=x_train_source, y=y_train_source, batch_size=batch_size, shuffle=False, sampler=sampler)
        alpha = class_sample_count[0]/(class_sample_count[1] + class_sample_count[0])
        criterion = m.FocalLoss(alpha=alpha, gamma=2)

        source_valid = utils.create_dataset(x=x_val_source, y=y_val_source, batch_size=batch_size, shuffle=False)
        dataloader_source = {'train': source_train, 'val': source_valid}

        x_target = pd.read_csv("./Datasets/processedData" + '/' + args.drug + "/target_data/target_scaled.csv", index_col=0)
        y_target = pd.read_csv("./Datasets/processedData" + '/' + args.drug + "/target_data/target_meta_data.csv", index_col=0)
        x_target = x_target.T
        y_target = y_target.reindex(x_target.index)
        x_train_target, x_val_target, y_train_target, y_val_target = train_test_split(x_target, y_target, test_size=0.2, random_state=i, stratify=y_target)
        x_train_target = x_train_target.T
        x_val_target = x_val_target.T

        random.seed(i)
        sample_0_train = random.sample(y_train_target[y_train_target["response"] == 0].index.tolist(), args.n)
        sample_1_train = random.sample(y_train_target[y_train_target["response"] == 1].index.tolist(), args.n)
        y_train_labeled = y_train_target.loc[sample_0_train + sample_1_train]
        x_train_labeled = x_train_target.loc[:, y_train_labeled.index.tolist()]
        y_train_unlabeled = y_train_target.drop(sample_0_train + sample_1_train, axis=0)
        x_train_unlabeled = x_train_target.loc[:, y_train_unlabeled.index.tolist()]

        random.seed(i)
        sample_0_val = random.sample(y_val_target[y_val_target["response"] == 0].index.tolist(), 25)
        sample_1_val = random.sample(y_val_target[y_val_target["response"] == 1].index.tolist(), 25)
        y_val_labeled = y_val_target.loc[sample_0_val + sample_1_val]
        x_val_labeled = x_val_target.loc[:, y_val_labeled.index.tolist()]
        y_val_unlabeled = y_val_target.drop(sample_0_val + sample_1_val, axis=0)
        x_val_unlabeled = x_val_target.loc[:, y_val_unlabeled.index.tolist()]

        target_train_unlabeled = utils.create_dataset(x=x_train_unlabeled, y=y_train_unlabeled, batch_size=batch_size, shuffle=True)
        target_valid_unlabeled = utils.create_dataset(x=x_val_unlabeled, y=y_val_unlabeled, batch_size=batch_size, shuffle=False)
        dataloader_unlabeled_target = {'train': target_train_unlabeled, 'val': target_valid_unlabeled}

        target_train_labeled = utils.create_dataset(x=x_train_labeled, y=y_train_labeled, batch_size=batch_size, shuffle=True)
        target_valid_labeled = utils.create_dataset(x=x_val_labeled, y=y_val_labeled, batch_size=batch_size, shuffle=False)
        dataloader_labeled_target = {'train': target_train_labeled, 'val': target_valid_labeled}

        y_test_target = y_target.drop(y_train_labeled.index.union(y_val_labeled.index), axis=0)
        x_target = x_target.T
        x_test_target = x_target.loc[:, y_test_target.index.tolist()]
        batch_size_test = len(x_test_target.T)
        test_dataset = utils.create_dataset(x=x_test_target, y=y_test_target, batch_size=batch_size_test, shuffle=False)

        bulk_tasks, sc_tasks = utils.cell_dim(drug=args.drug, gene=args.gene)
        dim_model_out = 2

        for path in [args.sc_all + args.drug, args.result + '/' + args.method]:
            if not os.path.exists(path):
                os.makedirs(path)
                print("The new directory is created!")

        sc_all_path = args.sc_all + args.drug + '/' + para
        result_path = args.result + args.method + '/' + args.drug + '/'

        if (args.device == "gpu"):
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            torch.cuda.set_device(device)
        else:
            device = 'cpu'
        print("device:", device)

        encoder = m.DAE(input_dim=sc_tasks['pathway'], fc_dim=256, AE_input_dim=bulk_tasks['expression'], AE_h_dims=encoder_hdims, pretrained_weights=None,drop=dropout)
        encoder.to(device)

        predictor = m.Predictor(input_dim=sc_tasks["pathway"], output_dim=32, drop_out=dropout)
        predictor.to(device)

        Predictor_adentropy = m.Predictor_adentropy(num_class=dim_model_out, inc=32)
        Predictor_adentropy.to(device)

        ppc = m.ProtoClassifier(size=2)
        ppc.to(device)

        loss_c = nn.CrossEntropyLoss()
        loss_e = nn.MSELoss()
        optimizer_d = torch.optim.Adagrad(itertools.chain(encoder.parameters(), predictor.parameters(), Predictor_adentropy.parameters()), lr=lr)
        encoder_f, predictor_f, Predictor_adentropy_f = trainer.train_semi_dae(encoder, predictor, Predictor_adentropy, ppc,
                                                                               dataloader_source, dataloader_unlabeled_target, dataloader_labeled_target,
                                                                               args.method, optimizer_d, loss_c, loss_e, criterion, epochs,
                                                                               writer=writer, device=device, auc_path=result_path)
        torch.save({'encoder_state_dict': encoder_f.state_dict(),
                         'predictor_state_dict': predictor_f.state_dict(),
                         'Predictor_adentropy_state_dict': Predictor_adentropy_f.state_dict()
                        }, sc_all_path + '.pth')

        print("Sampling Method:", args.sampling_method)
        print("Method:", args.method)
        print("Shot-Method:", str(args.n) + "-shot")
        print("Drug:", args.drug)
        loss_class = nn.CrossEntropyLoss()
        print("Transfer " + args.method + " finished")
        test_model = m.Test_Double_Model(predictor=predictor_f, encoder=encoder_f, adentropy_p=Predictor_adentropy_f)
        test_file = "./Datasets/processedData" + '/' + args.drug + '/' + str(args.n) + '_attention4_FocalLoss_threshold_0.75_0.55loss_0.40mme_0.02ppc_25val.tsv'
        trainer.test_shot(test_model, test_dataset, loss_class, device, test_file, i)

    test_file = "./Datasets/processedData" + '/' + args.drug + '/' + str(args.n) + '_attention4_FocalLoss_threshold_0.75_0.55loss_0.40mme_0.02ppc_25val.tsv'
    index_value = pd.read_csv(test_file, sep='\t', index_col=None, header=None)
    AUC_mean, AUC_std = index_value.iloc[:,3].mean(), index_value.iloc[:,3].std()
    AUPR_mean, AUPR_std = index_value.iloc[:, 5].mean(), index_value.iloc[:, 5].std()
    F1_mean, F1_std = index_value.iloc[:, 7].mean(), index_value.iloc[:, 7].std()
    ACC_mean, ACC_std = index_value.iloc[:, 9].mean(), index_value.iloc[:, 9].std()
    Recall_mean, Recall_std = index_value.iloc[:, 11].mean(), index_value.iloc[:, 11].std()
    Precision_mean, Precision_std = index_value.iloc[:, 13].mean(), index_value.iloc[:, 13].std()
    print("All Performance -> AUC:{:.4f}±{:.4f} AUPR:{:.4f}±{:.4f} F1_score:{:.4f}±{:.4f} ACC:{:.4f}±{:.4f} Recall:{:.4f}±{:.4f} Precision:{:.4f}±{:.4f}".format(
         AUC_mean, AUC_std, AUPR_mean, AUPR_std, F1_mean, F1_std, ACC_mean, ACC_std, Recall_mean, Recall_std, Precision_mean, Precision_std))
    test = "All Index"
    with open(test_file, 'a') as p:
        p.write("num\t{}\tAUC\t{:.4f}±{:.4f}\tAUPR\t{:.4f}±{:.4f}\tF1_score\t{:.4f}±{:.4f}\tACC\t{:.4f}±{:.4f}\tRecall\t{:.4f}±{:.4f}\tPrecision\t{:.4f}±{:.4f}\n".format(
                test, AUC_mean, AUC_std, AUPR_mean, AUPR_std, F1_mean, F1_std, ACC_mean, ACC_std, Recall_mean, Recall_std, Precision_mean, Precision_std))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='./Datasets/processedData', help='Path of the dataset used for model training')
    parser.add_argument('--drug', type=str, default='Erlotinib_GSE149214', help='choose drug,Can be Erlotinib,Dasatinib,Etoposide,Palbociclib and Ibrutinib_GSE111014.')
    parser.add_argument('--n', type=int, default=15, help='Number of shot. Default: 15')
    parser.add_argument('--gene', type=str, default="HVGs", help='choose data type. Can be all or HVGs. Default: HVGs')
    parser.add_argument('--sampling_method', type=str, default="weight", help='choose sampling type. Can be weight or smote. Default: weight')
    parser.add_argument('--shot_method', type=str, default="15-shot", help='choose shot type. Can be n-shot. Default: 15-shot')
    parser.add_argument('--result', type=str, default='save/results/sc/', help='Path of the training result report files')
    parser.add_argument('--sc_all', type=str, default='save/sc/all_path/', help='Path of the model in the sc level')
    parser.add_argument('--log_dir', type=str, default="./runsResult/version_Erlotinib_GSE149214", help='')
    parser.add_argument('--device', type=str, default="gpu", help='Device to train the model. Can be cpu or gpu. Default: gpu')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate of model training. Default: 1e-2')
    parser.add_argument('--epochs', type=int, default=300, help='Number of epoches training. Default: 300')
    parser.add_argument('--batch_size', type=int, default=200, help='Number of batch size when training. Default: 200')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout of neural network. Default: 0.3')
    parser.add_argument('--fix_source', type=int, default=0, help='Fix the parameters in the bulk model. 0: do not freeze, 1: freeze. Default: 0')
    parser.add_argument('--method', type=str, default="adv", help='choose model type. Can be DANN or CDAN. Default: DANN')
    parser.add_argument('--encoder_h_dims', type=str, default="512,256", help='Shape of the encoder. Each number represent the number of neuron in a layer. Layers are seperated by a comma. Default: 512,256')
    parser.add_argument('--bottleneck', type=int, default=256, help='Size of the bottleneck layer of the model. Default: 128')
    parser.add_argument('--predictor_h_dims', type=str, default="64,32", help='Shape of the predictor. Each number represent the number of neuron in a layer. Layers are seperated by a comma. Default: 128')
    parser.add_argument('--load_source_model', type=int, default=0, help='Load a trained bulk level or not. 0: do not load, 1: load. Default: 0')
    parser.add_argument('--printpathway', type=str, default='T', help='Print the cirtical pathway list: T: print. Default: T')
    warnings.filterwarnings("ignore")
    args, unknown = parser.parse_known_args()
    matplotlib.use('Agg')

    main(args)