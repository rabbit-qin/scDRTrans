import os
import numpy as np
import torch
from utils import save_AUCs
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from torch import nn
import utils
import torch.nn.functional as F


def test_shot(net, dataset, loss_c, device, test_file, i):
    net.eval()
    Y_pre = []
    Y_true = []
    AUC_scores = []
    APR_scores = []
    F1_scores = []
    ACC_score = []
    Recall_score = []
    Precision_score = []
    running_loss = []
    correct = 0

    for _, data in enumerate(zip(dataset)):
        data = data[0]
        x = data[0].to(device)
        y_true = data[1].to(device)
        y_out = net(x)
        loss = loss_c(y_out, y_true.long())
        running_loss.append(loss.item())
        y_out = nn.Softmax(dim=1)(y_out)

        y_true = y_true.cpu()
        y_out = y_out.cpu()
        y_score = y_out[:, 1]
        pred = y_out.max(1, keepdim=True)[1]
        correct += pred.eq(y_true.view_as(pred)).sum().item()

        AUC_train = utils.roc_auc_score_trainval(y_true.detach().numpy(), y_score.detach().numpy())
        APR_train = average_precision_score(y_true.detach().numpy(), y_score.detach().numpy())
        y_pre_label = pred.detach().numpy().flatten()
        F1_train = f1_score(y_true.detach().numpy(), y_pre_label)
        ACC = accuracy_score(y_true.detach().numpy(), y_pre_label)
        recall = recall_score(y_true.detach().numpy(), y_pre_label)
        precision = precision_score(y_true.detach().numpy(), y_pre_label)

        Y_pre.append(y_score)
        Y_true.append(y_true)

        AUC_scores.append(AUC_train)
        APR_scores.append(APR_train)
        F1_scores.append(F1_train)
        ACC_score.append(ACC)
        Recall_score.append(recall)
        Precision_score.append(precision)

    Y_pre = torch.cat(Y_pre, dim=0).detach().numpy()
    Y_true = torch.cat(Y_true, dim=0).detach().numpy()
    auc = np.mean(np.array(AUC_scores))
    aupr = np.mean(np.array(APR_scores))
    f1 = np.mean(np.array(F1_scores))
    ACC = np.mean(np.array(ACC_score))
    Recall = np.mean(np.array(Recall_score))
    Precision = np.mean(np.array(Precision_score))
    loss = np.mean(np.array(running_loss))
    print("Test Performance -> Loss:{:.6f} AUC:{:.4f} AUPR:{:.4f} F1_score:{:.4f} ACC:{:.4f} Recall:{:.4f} Precision:{:.4f} Accuracy: {}/{} ({:.0f}%) ".format(
        loss, auc, aupr, f1, ACC, Recall, Precision, correct, len(dataset.dataset), 100. * correct / len(dataset.dataset)))

    with open(test_file, 'a') as p:
        p.write("num\t{}\tAUC\t{}\tAUPR\t{}\tF1_score\t{}\tACC\t{}\tRecall\t{}\tPrecision\t{}\n".format(i, auc, aupr, f1, ACC, Recall, Precision))


def train_semi_dae(encoder, predictor, adentropy_p, ppc, source_loader, unlabeled_loader, labeled_loader, method,
                   optimizer, loss_class, loss_e, criterion, n_epochs, writer="None", device="cuda", auc_path=''):
    best_loss = np.inf
    best_auc = float('-inf')

    if not os.path.exists(auc_path):
        os.makedirs(auc_path)
    file_AUCs = auc_path + '/' + str(n_epochs) + '_train_AUCs_' + '.txt'

    AUCs = ('Epoch\tloss\tloss_dann\tloss_c\tloss_dae_s\tloss_dae_t\tphase\tauc\taupr\tF1_score\tACC\tRecall')
    with open(file_AUCs, 'w') as f:
        f.write(AUCs + '\n')

    for epoch in range(n_epochs):
        AUCtrain_source = []
        AUCval_source = []
        APRtrain_source = []
        APRval_source = []
        F1train_score = []
        F1val_score = []
        ACCtrain_score = []
        ACCval_score = []
        recalltrain_score = []
        recallval_score = []
        precisiontrain_score = []
        precisionval_score = []
        xt_unl_encoder = []
        xt_unl_predictor = []

        for phase in ['train', 'val']:
            if phase == 'train':
                encoder.train()
                predictor.train()
                adentropy_p.train()
            else:
                encoder.eval()
                predictor.eval()
                adentropy_p.eval()

            running_loss_train = []
            running_loss_valid = []
            running_dann_train = []
            running_dann_valid = []
            running_c_train = []
            running_c_valid = []
            running_e_train_s = []
            running_e_valid_s = []
            running_e_train_t = []
            running_e_valid_t = []
            running_soft_loss = []
            loss_c_xs_loss = []

            len_source = len(source_loader[phase])
            len_unlabeled = len(unlabeled_loader[phase])
            len_labeled = len(labeled_loader[phase])
            num_iter = max(len_source, len_unlabeled, len_labeled)

            for batch_idx in range(num_iter):
                if batch_idx % len_source == 0:
                    iter_source = iter(source_loader[phase])
                if batch_idx % len_unlabeled == 0:
                    iter_target_unl = iter(unlabeled_loader[phase])
                if batch_idx % len_labeled == 0:
                    iter_target = iter(labeled_loader[phase])
                xs, ys = iter_source.__next__()
                xt, yt = iter_target.__next__()
                xt_unl, _ = iter_target_unl.__next__()
                xt_unl = xt_unl.to(device)
                xs = xs.to(device)
                ys = ys.to(device)
                xt = xt.to(device)
                yt = yt.to(device)

                xs.requires_grad_(True)
                xt.requires_grad_(True)
                data = torch.cat((xs, xt), 0)
                target = torch.cat((ys, yt), 0)

                feature, ae_output = encoder(data)
                output_x = predictor(feature)
                output_2 = adentropy_p(output_x)
                loss_c_xt = loss_class(output_2[xs.size(0):(xs.size(0)+xt.size(0)),], yt.long())
                softmax_output = nn.Softmax(dim=1)(output_2)

                source_logits = output_2[0:xs.size(0), ]
                source_binary_logit = source_logits[:, 1] - source_logits[:, 0]

                if phase == 'train' and epoch > 99 and ppc.center is not None:
                    xs_fea = output_x.narrow(0, 0, xs.size(0))
                    sy2 = ppc(xs_fea.detach(), 0.6)  # T=0.6

                    log_softmax_out = F.log_softmax(source_logits, dim=1)
                    l_loss = criterion(source_binary_logit, ys)

                    soft_loss = -(sy2 * log_softmax_out).sum(axis=1)
                    loss_c_xs = ((1 - 0.40) * l_loss + 0.40 * soft_loss).mean()   # alpha=0.40
                    running_soft_loss.append(soft_loss.mean().item())
                else:
                    loss_c_xs = criterion(source_binary_logit, ys)

                loss_c_xs_loss.append(loss_c_xs.item())
                loss_c = loss_c_xs + loss_c_xt

                loss_ae_t = loss_e(xt, ae_output.narrow(0, xs.size(0), xt.size(0)))
                loss_ae_s = loss_e(xs, ae_output.narrow(0, 0, xs.size(0)))

                loss = loss_c + 0.55*(loss_ae_t + loss_ae_s)

                if phase == 'train':
                    loss.backward(retain_graph=True)
                    optimizer.step()
                    optimizer.zero_grad()

                output_unl, ae_output = encoder(xt_unl)
                output_unl = predictor(output_unl)

                if phase == 'train':
                    ppc_feature = output_unl.detach()
                    ppc_prediction = adentropy_p(ppc_feature).detach()
                    loss_t = utils.adentropy(adentropy_p, output_unl, 0.02)
                    loss_t.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                else:
                    with torch.no_grad():
                        loss_t = utils.adentropy(adentropy_p, output_unl, 0.02)

                if phase == 'train':
                    xt_unl_encoder.append(ppc_feature)
                    xt_unl_predictor.append(ppc_prediction)

                if phase == 'train':
                    yst = torch.cat([ys.cpu(),yt.cpu()], dim=0)
                    y_pre = softmax_output.narrow(0, 0, xs.size(0)+xt.size(0)).cpu()
                    y_pre = y_pre[:, 1]
                    AUC_train = utils.roc_auc_score_trainval(yst.detach().numpy(), y_pre.detach().numpy())
                    APR_train = average_precision_score(yst.detach().numpy(), y_pre.detach().numpy())

                    y_out = softmax_output.narrow(0, 0, xs.size(0)+xt.size(0)).cpu()
                    pred = y_out.max(1, keepdim=True)[1]
                    y_pre_label = pred.detach().numpy().flatten()
                    F1_train = f1_score(yst.detach().numpy(), y_pre_label)
                    ACC = accuracy_score(yst.detach().numpy(), y_pre_label)
                    recall = recall_score(yst.detach().numpy(), y_pre_label)
                    precision = precision_score(yst.detach().numpy(), y_pre_label)
                    AUCtrain_source.append(AUC_train)
                    APRtrain_source.append(APR_train)
                    F1train_score.append(F1_train)
                    ACCtrain_score.append(ACC)
                    recalltrain_score.append(recall)
                    precisiontrain_score.append(precision)

                    running_loss_train.append(loss.item())
                    running_dann_train.append(-loss_t.item())
                    running_c_train.append(loss_c.item())
                    running_e_train_s.append(loss_ae_s.item())
                    running_e_train_t.append(loss_ae_t.item())

                if phase == 'val':
                    yst = torch.cat([ys.cpu(), yt.cpu()], dim=0)
                    y_pre = softmax_output.narrow(0, 0, xs.size(0) + xt.size(0)).cpu()
                    y_pre = y_pre[:, 1]
                    AUC_val = utils.roc_auc_score_trainval(yst.detach().numpy(), y_pre.detach().numpy())
                    APR_val = average_precision_score(yst.detach().numpy(), y_pre.detach().numpy())

                    y_out = softmax_output.narrow(0, 0, xs.size(0) + xt.size(0)).cpu()
                    pred = y_out.max(1, keepdim=True)[1]
                    y_pre_label = pred.detach().numpy().flatten()
                    F1_val = f1_score(yst.detach().numpy(), y_pre_label)
                    ACC = accuracy_score(yst.detach().numpy(), y_pre_label)
                    recall = recall_score(yst.detach().numpy(), y_pre_label)
                    precision = precision_score(yst.detach().numpy(), y_pre_label)
                    AUCval_source.append(AUC_val)
                    APRval_source.append(APR_val)
                    F1val_score.append(F1_val)
                    ACCval_score.append(ACC)
                    recallval_score.append(recall)
                    precisionval_score.append(precision)

                    running_loss_valid.append(loss.item())
                    running_dann_valid.append(-loss_t.item())
                    running_c_valid.append(loss_c.item())
                    running_e_valid_s.append(loss_ae_s.item())
                    running_e_valid_t.append(loss_ae_t.item())

            if phase == 'train':
                epoch_loss = np.mean(np.array(running_loss_train))
                epoch_dann = np.mean(np.array(running_dann_train))
                epoch_c = np.mean(np.array(running_c_train))
                epoch_ae_s = np.mean(np.array(running_e_train_s))
                epoch_ae_t = np.mean(np.array(running_e_train_t))
                epoch_soft_loss = np.mean(np.array(running_soft_loss))
                epoch_loss_c_xs = np.mean(loss_c_xs_loss)
                auc = np.mean(np.array(AUCtrain_source))
                aupr = np.mean(np.array(APRtrain_source))
                F1_score = np.mean(np.array(F1train_score))
                accuracy = np.mean(np.array(ACCtrain_score))
                recall = np.mean(np.array(recalltrain_score))
                precision = np.mean(np.array(precisiontrain_score))

                writer.add_scalar('train/epoch_loss_all', epoch_loss, epoch)
                writer.add_scalar('train/epoch_mme', epoch_dann, epoch)
                writer.add_scalar('train/epoch_ce', epoch_c, epoch)
                writer.add_scalar('train/epoch_ae_s', epoch_ae_s, epoch)
                writer.add_scalar('train/epoch_ae_t', epoch_ae_t, epoch)
                writer.add_scalar('train/soft_loss', epoch_soft_loss, epoch)
                writer.add_scalar('train/loss_c_xs', epoch_loss_c_xs, epoch)

            if phase == 'val':
                epoch_loss = np.mean(np.array(running_loss_valid))
                epoch_dann = np.mean(np.array(running_dann_valid))
                epoch_c = np.mean(np.array(running_c_valid))
                epoch_ae_s = np.mean(np.array(running_e_valid_s))
                epoch_ae_t = np.mean(np.array(running_e_valid_t))
                auc = np.mean(np.array(AUCval_source))
                aupr = np.mean(np.array(APRval_source))
                F1_score = np.mean(np.array(F1val_score))
                accuracy = np.mean(np.array(ACCval_score))
                recall = np.mean(np.array(recallval_score))
                precision = np.mean(np.array(precisionval_score))

                writer.add_scalar('validation/epoch_loss_all', epoch_loss, epoch)
                writer.add_scalar('validation/epoch_mme', epoch_dann, epoch)
                writer.add_scalar('validation/epoch_ce', epoch_c, epoch)
                writer.add_scalar('validation/epoch_ae_s', epoch_ae_s, epoch)
                writer.add_scalar('validation/epoch_ae_t', epoch_ae_t, epoch)

            train_AUCs = [epoch, epoch_loss, epoch_dann, epoch_c, epoch_ae_s, epoch_ae_t, phase, auc, aupr, F1_score, accuracy, recall, precision]
            save_AUCs(train_AUCs, file_AUCs)

            if (phase == 'val') and (epoch_loss < best_loss or best_auc < auc):
                best_auc = auc
                best_loss = epoch_loss

        if epoch == 99:
            xt_unl_encoder = torch.cat(xt_unl_encoder, dim=0)
            xt_unl_predictor = torch.cat(xt_unl_predictor, dim=0)
            ppc.init(xt_unl_predictor, xt_unl_encoder, threshold=0.75)
        if epoch > 99 and (epoch+1) % 20 == 0:
            xt_unl_encoder = torch.cat(xt_unl_encoder, dim=0)
            xt_unl_predictor = torch.cat(xt_unl_predictor, dim=0)
            ppc.init(xt_unl_predictor, xt_unl_encoder, threshold=0.75)

    return encoder, predictor, adentropy_p


def train_baseline_dae(encoder, predictor, adentropy_p, source_loader, method, optimizer, loss_class, loss_e, criterion,
                       n_epochs, start_epoch=0, save_path="save/model.pkl", device="cuda", auc_path=''):
    best_loss = np.inf
    best_auc = float('-inf')

    if not os.path.exists(auc_path):
        os.makedirs(auc_path)
    file_AUCs = auc_path + '/' + '_train_AUCs_' + '.txt'

    AUCs = ('Epoch\tloss\tloss_dann\tloss_c\tloss_dae\tphase\tauc\taupr')
    with open(file_AUCs, 'w') as f:
        f.write(AUCs + '\n')

    for epoch in range(n_epochs):

        AUCtrain_source = []
        AUCval_source = []

        APRtrain_source = []
        APRval_source = []

        for phase in ['train', 'val']:
            if phase == 'train':
                encoder.train()
                predictor.train()
                adentropy_p.train()
            else:
                encoder.eval()
                predictor.eval()
                adentropy_p.eval()

            running_loss_train = []
            running_loss_valid = []
            running_dann_train = []
            running_dann_valid = []
            running_c_train = []
            running_c_valid = []
            running_e_train = []
            running_e_valid = []

            len_source = len(source_loader[phase])
            iter_source = iter(source_loader[phase])
            for batch_idx in range(len_source):
                xs, ys = next(iter_source)
                xs = xs.to(device)
                ys = ys.to(device)
                xs.requires_grad_(True)
                feature, ae_output = encoder(xs)

                output = predictor(feature)
                output = adentropy_p(output)

                binary_logit = output[:, 1] - output[:, 0]
                loss_c = criterion(binary_logit, ys)

                softmax_output = nn.Softmax(dim=1)(output)
                loss_ae = loss_e(xs, ae_output.narrow(0, 0, xs.size(0)))

                loss = loss_c + loss_ae

                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    ys = ys.cpu()
                    y_pre = softmax_output.narrow(0, 0, xs.size(0)).cpu()
                    y_pre = y_pre[:, 1]
                    AUC_train = utils.roc_auc_score_trainval(ys.detach().numpy(), y_pre.detach().numpy())
                    APR_train = average_precision_score(ys.detach().numpy(), y_pre.detach().numpy())
                    AUCtrain_source.append(AUC_train)
                    APRtrain_source.append(APR_train)
                    running_loss_train.append(loss.item())
                    running_c_train.append(loss_c.item())
                    running_e_train.append(loss_ae.item())
                if phase == 'val':
                    ys = ys.cpu()
                    y_pre = softmax_output.narrow(0, 0, xs.size(0)).cpu()
                    y_pre = y_pre[:, 1]
                    AUC_val = utils.roc_auc_score_trainval(ys.detach().numpy(), y_pre.detach().numpy())
                    APR_val = average_precision_score(ys.detach().numpy(), y_pre.detach().numpy())
                    AUCval_source.append(AUC_val)
                    APRval_source.append(APR_val)
                    running_loss_valid.append(loss.item())
                    running_c_valid.append(loss_c.item())
                    running_e_valid.append(loss_ae.item())

            if phase == 'train':
                epoch_loss = np.mean(np.array(running_loss_train))
                epoch_dann = np.mean(np.array(running_dann_train))
                epoch_c = np.mean(np.array(running_c_train))
                epoch_ae = np.mean(np.array(running_e_train))
                auc = np.mean(np.array(AUCtrain_source))
                aupr = np.mean(np.array(APRtrain_source))

            if phase == 'val':
                epoch_loss = np.mean(np.array(running_loss_valid))
                epoch_dann = np.mean(np.array(running_dann_valid))
                epoch_c = np.mean(np.array(running_c_valid))
                epoch_ae = np.mean(np.array(running_e_valid))
                auc = np.mean(np.array(AUCval_source))
                aupr = np.mean(np.array(APRval_source))

            train_AUCs = [epoch, epoch_loss, epoch_dann, epoch_c, epoch_ae, phase, auc, aupr]
            print("Epoch:{} Phase:{} AUC: {:.3f} AUPR: {:.3f} ".format(epoch, phase, auc, aupr))
            save_AUCs(train_AUCs, file_AUCs)

            if (phase == 'val') and (epoch_loss < best_loss or best_auc < auc):
                best_auc = auc
                best_loss = epoch_loss
    return encoder, predictor, adentropy_p
