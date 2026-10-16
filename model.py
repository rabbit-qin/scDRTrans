import pandas as pd
import torch.nn as nn
import torch
import numpy as np
from copy import deepcopy
import torch.nn.functional as F


class Predictor(nn.Module):
    def __init__(self, input_dim, output_dim=32, drop_out=0.3):
        super(Predictor, self).__init__()
        modules = [nn.Sequential(
            nn.Linear(input_dim, output_dim),
            # nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Dropout(drop_out))]
        # in_channels = h_dim
        self.predictor = nn.Sequential(*modules)

    def forward(self, input):
        output = self.predictor(input)
        return output


class Predictor_adentropy(nn.Module):
    def __init__(self, num_class=2, inc=32, temp=0.05, *args, **kwargs):
        super(Predictor_adentropy, self).__init__()
        super().__init__(*args, **kwargs)
        self.fc = nn.Linear(inc, num_class, bias=False)
        self.num_class = num_class
        self.temp = temp

    def forward(self, x, reverse=False, eta=0.1):
        if reverse:
            x = grad_reverse(x, eta)
        x = F.normalize(x)
        x_out = self.fc(x) / self.temp
        return x_out


class MLP(nn.Module):
    def __init__(self, input_dim, latent_dim=128, h_dims=[512], drop_out=0.3):
        super(MLP, self).__init__()
        modules = []
        hidden_dims = deepcopy(h_dims)
        hidden_dims.insert(0, input_dim)
        self.__in_features = latent_dim

        for i in range(1, len(hidden_dims)):
            i_dim = hidden_dims[i - 1]
            o_dim = hidden_dims[i]
            modules.append(
                nn.Sequential(
                    nn.Linear(i_dim, o_dim),
                    nn.BatchNorm1d(o_dim),
                    nn.ReLU(),
                    nn.Dropout(drop_out)
                )
            )
        self.encoder = nn.Sequential(*modules)
        self.bottleneck = nn.Linear(hidden_dims[-1], latent_dim)

    def forward(self, input):
        embedding = self.encoder(input)
        output = self.bottleneck(embedding)
        return output

    def output_num(self):
        return self.__in_features

class AttentionBlock(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout):
        super().__init__()
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        assert hid_dim % n_heads == 0
        self.f_q = nn.Linear(hid_dim, hid_dim)
        self.f_k = nn.Linear(hid_dim, hid_dim)
        self.f_v = nn.Linear(hid_dim, hid_dim)
        self.fc = nn.Linear(hid_dim, hid_dim)
        self.do = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim // n_heads])).cuda()

    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]
        Q = self.f_q(query)
        K = self.f_k(key)
        V = self.f_v(value)
        Q = Q.view(batch_size, self.n_heads, self.hid_dim // self.n_heads).unsqueeze(3)
        K_T = K.view(batch_size, self.n_heads, self.hid_dim // self.n_heads).unsqueeze(3).transpose(2,3)
        V = V.view(batch_size, self.n_heads, self.hid_dim // self.n_heads).unsqueeze(3)
        energy = torch.matmul(Q, K_T) / self.scale
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)
        attention = self.do(F.softmax(energy, dim=-1))
        weighter_matrix = torch.matmul(attention, V)
        weighter_matrix = weighter_matrix.permute(0, 2, 1, 3).contiguous()
        weighter_matrix = weighter_matrix.view(batch_size, self.n_heads * (self.hid_dim // self.n_heads))
        weighter_matrix = self.do(self.fc(weighter_matrix))
        return weighter_matrix

class CrossAttentionBlock(nn.Module):
    def __init__(self, hid_dim):
        super(CrossAttentionBlock, self).__init__()
        self.att = AttentionBlock(hid_dim = hid_dim, n_heads = 1, dropout=0.3)
        self.layer_norm = nn.LayerNorm(hid_dim)

    def forward(self,embedding2d):
        embedding2d_in = embedding2d + self.att(embedding2d, embedding2d, embedding2d)
        return embedding2d_in


class AE(nn.Module):
    def __init__(self, input_dim, latent_dim=128, h_dims=[512], drop_out=0.3):
        super(AE, self).__init__()
        self.latent_dim = latent_dim

        modules = []
        hidden_dims = deepcopy(h_dims)
        hidden_dims.insert(0, input_dim)

        for i in range(1, len(hidden_dims)):
            i_dim = hidden_dims[i - 1]
            o_dim = hidden_dims[i]
            modules.append(
                nn.Sequential(
                    nn.Linear(i_dim, o_dim),
                    nn.BatchNorm1d(o_dim),
                    nn.ReLU(),
                    nn.Dropout(drop_out))
            )
        self.encoder = nn.Sequential(*modules)
        self.selfAttention01 = CrossAttentionBlock(hid_dim=256)
        self.selfAttention02 = CrossAttentionBlock(hid_dim=256)
        self.selfAttention03 = CrossAttentionBlock(hid_dim=256)
        self.selfAttention04 = CrossAttentionBlock(hid_dim=256)
        self.bottleneck = nn.Linear(hidden_dims[-1], latent_dim)

        modules = []
        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1])
        hidden_dims.reverse()
        for i in range(len(hidden_dims) - 2):
            modules.append(
                nn.Sequential(
                    nn.Linear(hidden_dims[i], hidden_dims[i + 1]),
                    nn.BatchNorm1d(hidden_dims[i + 1]),
                    nn.ReLU(),
                    nn.Dropout(drop_out))
            )
        self.decoder = nn.Sequential(*modules)
        self.final_layer = nn.Sequential(
            nn.Linear(hidden_dims[-2], hidden_dims[-1])
        )

    def encode(self, input):
        result = self.encoder(input)
        result = self.selfAttention01(result)
        result = self.selfAttention02(result)
        result = self.selfAttention03(result)
        result = self.selfAttention04(result)
        embedding = self.bottleneck(result)
        return embedding

    def decode(self, z):
        result = self.decoder_input(z)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def forward(self, input):
        embedding = self.encode(input)
        output = self.decode(embedding)
        return embedding, output


class DAE(nn.Module):
    def __init__(self, input_dim=128, fc_dim=256, AE_input_dim=17419, AE_h_dims=[512, 256], pretrained_weights=None, drop=0.1, act='relu'):
        super(DAE, self).__init__()
        if pretrained_weights is not None:
            self.load_state_dict((torch.load(pretrained_weights)))
        self.__in_features = input_dim
        self.ae = AE(input_dim=AE_input_dim, latent_dim=input_dim, h_dims=AE_h_dims, drop_out=drop)
        self.classifier = Predictor(input_dim=input_dim,output_dim=32)

    def dae(self, data):
        z = data
        if self.training:
            mask = torch.rand_like(z) < 0.2
            z = z.masked_fill(mask, 0.0)
        embedding = self.ae.encode(z)
        return embedding

    def forward(self, data):
        z = data
        if self.training:
            mask = torch.rand_like(z) < 0.2
            z = z.masked_fill(mask, 0.0)

        data.requires_grad_(True)
        embedding = self.ae.encode(z)
        ae_output = self.ae.decode(embedding)
        return embedding, ae_output

    def output_num(self):
        return self.__in_features


class ProtoClassifier(nn.Module):
    def __init__(self, size=2):
        super(ProtoClassifier, self).__init__()
        self.center = None
        self.label = None
        self.size = size

    def init(self, t_pred, t_feat, threshold):
        label = t_pred.argmax(dim=1)
        softmax_output = nn.Softmax(dim=1)(t_pred)
        pred_max = softmax_output.max(1)
        mask = pred_max[0] >= threshold
        t_feat = t_feat[mask]
        label = label[mask]
        print("labels_number: {}".format(label.shape))
        center = compute_weighted_prototypes(t_feat, label, self.size)

        class_count = torch.bincount(label, minlength=self.size)
        invalid_idx = class_count == 0
        if invalid_idx.any():
            if self.center is None:
                missing_classes = torch.where(invalid_idx)[0].tolist()
                print("PPC initialization skipped, missing classes: {}".format(missing_classes))
                return False
            center[invalid_idx] = self.center[invalid_idx]
        self.label = label

        self.center = center.detach()
        return True

    def forward(self, x, T=1.0):
        dist = torch.cdist(x, self.center)
        return F.softmax(-dist * T, dim=1)


def compute_weighted_prototypes(t_feat, labels, num_classes, eps=1e-8):
    centers = []
    for i in range(num_classes):
        mask = (labels == i)
        supp = t_feat[mask]
        S = supp.shape[0]
        if S <= 1:
            if S == 1:
                centers.append(supp.mean(dim=0))
            else:
                centers.append(torch.full((t_feat.shape[1],), float('nan'), device=t_feat.device))
            continue
        norms = torch.norm(supp, dim=1, keepdim=True) + eps
        normed = supp / norms
        dots = torch.mm(normed, normed.t())
        avg_cos_excl_self = (dots.sum(dim=1) - 1.0) / (S - 1)
        weighted_proto = torch.sum(avg_cos_excl_self.unsqueeze(1) * supp, dim=0) / S
        centers.append(weighted_proto)
    center = torch.nan_to_num(torch.vstack(centers), nan=0.0)
    return center


class Test_Double_Model(nn.Module):
    def __init__(self, predictor, encoder, adentropy_p):
        super(Test_Double_Model, self).__init__()
        self.encoder = encoder
        self.predictor = predictor
        self.adentropy_p = adentropy_p

    def forward(self, test_data):
        feature = self.encoder(test_data)[0]
        output = self.predictor(feature)
        output = self.adentropy_p(output)
        return output


from torch.autograd import Function

class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

def grad_reverse(x, lambd=1.0):
    return GradReverse.apply(x,lambd)


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        targets = targets.float()
        p = torch.sigmoid(inputs)
        bce_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        p_t = p * targets + (1 - p) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_loss = alpha_t * ((1 - p_t) ** self.gamma) * bce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss