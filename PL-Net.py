import argparse
import os
import pickle
import time, submitit, cv2, wandb, shutil, copy, math, json, random, argparse
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
import torchvision.transforms as transforms
from PIL import Image
from torch.utils import data
from torch.nn.modules.module import Module
from torchvision.utils import make_grid
from pathlib import Path
from tqdm import tqdm
from math import floor, log2
from model.resnet_generator_context import *
from bounding_box import bounding_box as bb
from torch.distributions.multivariate_normal import MultivariateNormal
from evaluate import calculate_scores_given_paths
from torchvision.ops.boxes import box_area
from gmm import GMM_head, Log_Pdf
from transformer import TransformerEncoder,RETransformerEncoder, TransformerDecoder, TransformerEncoderLayer,RETransformerEncoderLayer, TransformerDecoderLayer, PositionalEncoder
from itertools import permutations


class Dataset_JSON(data.Dataset):

    def __init__(self):
        super().__init__()

        if 'bird' in  args.dataset:
            self.data = np.load(args.data_path, allow_pickle=True) 
        
        elif 'creature' in  args.dataset:
            self.data = np.load(args.data_path, allow_pickle=True) 

        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):


        bbox = self.data[index]['bbox']
        intial_xy = self.data[index]['intial_xy']
        label = self.data[index]['label']
        raster = self.data[index]['raster']
        raster_initial = self.data[index]['raster_initial']
        text = self.data[index]['text']

        return raster, label, bbox, intial_xy, raster_initial, text


def collate_fn(batch):

    batch = list(filter(lambda x: x is not None, batch))

    max_len = max(list((batch[i][3].shape[0] for i in range(len(batch)))))

    for idx, bt in enumerate(batch):

        batch[idx] = [batch[idx][0],batch[idx][1],batch[idx][2],
                     np.concatenate([batch[idx][3], np.zeros((max_len-len(batch[idx][3]), 2))],0),
                     batch[idx][4],
                     ]


    return torch.utils.data.dataloader.default_collate(batch)


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
    
    
def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx    


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    if torch.cuda.is_available():
        return torch.sparse.FloatTensor(indices, values, shape).cuda()
    else:
        return torch.sparse.FloatTensor(indices, values, shape)

class GraphFunc(nn.Module):
    def __init__(self, z_dim):
        super(GraphFunc, self).__init__()
        """
        DeepSets Function
        """
        self.gc1 = GraphConvolution(z_dim, z_dim * 4)
        self.gc2 = GraphConvolution(z_dim * 4, z_dim)
        self.z_dim = z_dim

    def forward(self, graph_input_raw, graph_label):
        #import ipdb; ipdb.set_trace()
        """
        set_input, seq_length, set_size, dim
        """
        set_length, set_size, dim = graph_input_raw.shape
        assert(dim == self.z_dim)
        set_output_list = []
        
        for g_index in range(set_length):
            graph_input = graph_input_raw[g_index, :]
            # construct the adj matrix
            unique_class = np.unique(graph_label[g_index,:].cpu().numpy())
            edge_set = []
            for c in unique_class:
                current_index = np.where(graph_label[g_index,:].cpu().numpy() == c)[0].tolist()
                if len(current_index) > 1:
                    edge_set.append(np.array(list(permutations(current_index, 2))))
            
            if len(edge_set) == 0:
                adj = sp.coo_matrix((np.array([0]), (np.array([0]), np.array([0]))),
                                    shape=(graph_label[g_index,:].shape[0], graph_label[g_index,:].shape[0]),
                                    dtype=np.float32)
            else:
                edge_set = np.concatenate(edge_set, 0)
                adj = sp.coo_matrix((np.ones(edge_set.shape[0]), (edge_set[:, 0], edge_set[:, 1])),
                                    shape=(graph_label[g_index,:].shape[0], graph_label[g_index,:].shape[0]),
                                    dtype=np.float32)        
            adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
            adj = normalize(adj + sp.eye(adj.shape[0]))
            adj = sparse_mx_to_torch_sparse_tensor(adj)
            
            # do GCN process
            residual = graph_input
            graph_input = F.relu(self.gc1(graph_input, adj))
            graph_input = F.dropout(graph_input, 0.5, training=self.training)
            graph_input = self.gc2(graph_input, adj)        
            set_output = residual + graph_input
            set_output_list.append(set_output)
        
        return torch.stack(set_output_list)

def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)

def box_xywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [x_c, y_c, x_c + w, y_c + h]
    return torch.stack(b, dim=-1)

def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/
    The boxes should be in [x0, y0, x1, y1] format
    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area

class ScheduledOptim():
    '''A simple wrapper class for learning rate scheduling'''

    def __init__(self, optimizer, lr_mul, d_model, n_warmup_steps):
        self._optimizer = optimizer
        self.lr_mul = lr_mul
        self.d_model = d_model
        self.n_warmup_steps = n_warmup_steps
        self.n_steps = 0


    def step_and_update_lr(self):
        "Step with the inner optimizer"
        self._update_learning_rate()
        self._optimizer.step()


    def zero_grad(self):
        "Zero out the gradients with the inner optimizer"
        self._optimizer.zero_grad()


    def _get_lr_scale(self):
        d_model = self.d_model
        n_steps, n_warmup_steps = self.n_steps, self.n_warmup_steps
        return (d_model ** -0.5) * min(n_steps ** (-0.5), n_steps * n_warmup_steps ** (-1.5))


    def _update_learning_rate(self):
        ''' Learning rate scheduling per step '''

        self.n_steps += 1
        lr = self.lr_mul * self._get_lr_scale()

        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr

def lr_decay(optimizer):
    """Decay learning rate by a factor of lr_decay"""
    for param_group in optimizer.param_groups:
        if param_group['lr']>0.00001:
            param_group['lr'] *= 0.9999
    return optimizer

def fix_bboxs(bbox, a, b):

    bs, N, _ = bbox.shape

    bbox = bbox.view(bs*N, -1)
    a = a.view(bs*N, -1)
    b = b.view(bs*N, -1)
    
    for k in range(bbox.shape[0]):

        if bbox[k][0] < 0:
            a[k] = torch.Tensor([-0.6, -0.6, 0.5, 0.5]).to('cuda')
            b[k] = torch.Tensor([-0.6, -0.6, 0.5, 0.5]).to('cuda')

    return a.view(bs, N, -1), b.view(bs, N, -1)


def visalize_bboxs(dataset, inp, label, bbox):

    sc = 1

    inp = np.ones((128*sc, 128*sc, 3))*255

    for i in range(len(label)):

        if label[i] != 0 and id_to_part[label[i]-1] != 'initial':

            x, y, w, h = bbox[i]*128*sc

            color = list(reversed([int(j) for j in color_table[id_to_part[label[i]-1]]*255]))
            lbl = id_to_part[label[i]-1]
            x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)

            inp = cv2.rectangle(inp, (x1, y1), (x2, y2), (209, 240, 251), -1)
 

    for i in range(len(label)):

        if label[i] != 0 and id_to_part[label[i]-1] != 'initial':

            x, y, w, h = bbox[i]*128*sc

            color = list(reversed([int(j) for j in color_table[id_to_part[label[i]-1]]*255]))
            lbl = id_to_part[label[i]-1]
            x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)

            inp = cv2.rectangle(inp, (x1, y1), (x2, y2), color, 2)


    return inp

class KLLoss(nn.Module):
    def __init__(self, kl_tolerance = 0.0):
        super(KLLoss, self).__init__()
        self.kl_tolerance = torch.tensor(kl_tolerance)

    '''
    Input:
        mus[batch, latent_size]:
        sigmas[batch, latent_size]:
    '''
    def forward(self, mus, sigmas):
        loss = - (0.5) * torch.mean(1 + torch.log(sigmas)*2.0  - mus*mus - sigmas*sigmas)
        return torch.max(loss, self.kl_tolerance.to(loss.device))


class KLDLossNoReduction(nn.Module):
    def forward(self, mu1, logvar1, mu2, logvar2):

        sigma1 = logvar1.mul(0.5).exp()
        sigma2 = logvar2.mul(0.5).exp()
        kld = torch.log(sigma2/sigma1+1e-8) + (torch.exp(logvar1) + (mu1 - mu2)**2)/(2*torch.exp(logvar2)+1e-8) - 1/2
        return kld.mean()# / BATCH_SIZE


class KLDtorch(nn.Module):
    def forward(self, mu1, logvar1, mu2, logvar2):

        std1 = F.softplus(logvar1)
        std2 = F.softplus(logvar2)

        distribution1 = torch.distributions.Normal(mu1, std1)
        distribution2 = torch.distributions.Normal(mu2, std2)
        distribution1_fix = torch.distributions.Normal(mu1.detach(), std1.detach())
        distribution2_fix = torch.distributions.Normal(mu2.detach(), std2.detach())
        distributionN = torch.distributions.Normal(torch.zeros_like(mu1), torch.ones_like(std1))


def compute_kernel(x, y):
    x_size = x.size()[0]
    y_size = y.size()[0]
    dim = x.size()[1]

    tiled_x = x.unsqueeze(1).repeat(1, y_size, 1)
    tiled_y = y.unsqueeze(0).repeat(x_size, 1, 1)
    return torch.exp(-torch.mean((tiled_x - tiled_y)**2, dim=2) / float(dim))

def mmd_loss(x, y):
    x_kernel = compute_kernel(x, x)
    y_kernel = compute_kernel(y, y)
    xy_kernel = compute_kernel(x, y)
    return torch.mean(x_kernel) + torch.mean(y_kernel) - 2 * torch.mean(xy_kernel)
    
class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

def BoxRelationalEmbedding(f_g, dim_g=64, wave_len=1000, trignometric_embedding=True):
    """
    Given a tensor with bbox coordinates for detected objects on each batch image,
    this function computes a matrix for each image
    with entry (i,j) given by a vector representation of the
    displacement between the coordinates of bbox_i, and bbox_j
    input: np.array of shape=(batch_size, max_nr_bounding_boxes, 4)
    output: np.array of shape=(batch_size, max_nr_bounding_boxes, max_nr_bounding_boxes, 64)
    """
    # returns a relational embedding for each pair of bboxes, with dimension = dim_g
    # follow implementation of https://github.com/heefe92/Relation_Networks-pytorch/blob/master/model.py#L1014-L1055

    f_g = f_g.cuda()

    batch_size = f_g.size(0)

    x_min, y_min, x_max, y_max = torch.chunk(f_g, 4, dim=-1)

    cx = (x_min + x_max) * 0.5
    cy = (y_min + y_max) * 0.5
    w = (x_max - x_min) + 1.
    h = (y_max - y_min) + 1.

    # cx.view(1,-1) transposes the vector cx, and so dim(delta_x) = (dim(cx), dim(cx))
    delta_x = cx - cx.view(batch_size, 1, -1)
    delta_x = torch.clamp(torch.abs(delta_x / w), min=1e-3)
    delta_x = torch.log(delta_x)

    delta_y = cy - cy.view(batch_size, 1, -1)
    delta_y = torch.clamp(torch.abs(delta_y / h), min=1e-3)
    delta_y = torch.log(delta_y)

    delta_w = torch.log(w / w.view(batch_size, 1, -1))
    delta_h = torch.log(h / h.view(batch_size, 1, -1))

    matrix_size = delta_h.size()
    delta_x = delta_x.view(batch_size, matrix_size[1], matrix_size[2], 1)
    delta_y = delta_y.view(batch_size, matrix_size[1], matrix_size[2], 1)
    delta_w = delta_w.view(batch_size, matrix_size[1], matrix_size[2], 1)
    delta_h = delta_h.view(batch_size, matrix_size[1], matrix_size[2], 1)

    position_mat = torch.cat((delta_x, delta_y, delta_w, delta_h), -1)

    if trignometric_embedding == True:
        feat_range = torch.arange(dim_g / 8).cuda()
        dim_mat = feat_range / (dim_g / 8)
        dim_mat = 1. / (torch.pow(wave_len, dim_mat))

        dim_mat = dim_mat.view(1, 1, 1, -1)
        position_mat = position_mat.view(batch_size, matrix_size[1], matrix_size[2], 4, -1)
        position_mat = 100. * position_mat

        mul_mat = position_mat * dim_mat
        mul_mat = mul_mat.view(batch_size, matrix_size[1], matrix_size[2], -1)
        sin_mat = torch.sin(mul_mat)
        cos_mat = torch.cos(mul_mat)
        embedding = torch.cat((sin_mat, cos_mat), -1)
    else:
        embedding = position_mat

    s1, s2, s3, s4 = embedding.size()
    embds = torch.zeros((s1, s2+1, s3+1, s4)).cuda()
    embds[:,1:,1:,:] = embedding

    return(embds)

class Generator(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=True,
                 return_intermediate_dec=False):
        super(Generator, self).__init__()


        encoder_layer = RETransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.trencoder_1 = RETransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)


        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.trencoder_2 = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.trdecoder_1 = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)


        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.trdecoder_2 = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)

        self.PositionalEncoder = PositionalEncoder(d_model, max_seq_len = 50)

        self.netEemb = nn.Embedding(1, d_model).weight
        self.netFemb = nn.Embedding(1, d_model).weight
        self.part_emb = nn.Embedding(N_PARTS+1, d_model).weight
        self.ab_layer = nn.Linear(2, d_model)
        self.intial_xy_layer = nn.Linear(2, d_model)
        self.intial_xy_layer_dec = nn.Linear(2, d_model)
        self.z_layer = nn.Linear(2*d_model, d_model)

        self._4tod_fc = nn.Linear(4, d_model)

        self.merge_linear = MLP(2*d_model, d_model, d_model, 3)

        self.linear_mu1 = nn.Linear(2*d_model, d_model)
        self.linear_mu2 = nn.Linear(d_model, d_model)
        self.linear_logvar1 = nn.Linear(2*d_model, d_model)
        self.linear_logvar2 = nn.Linear(d_model, d_model)

        self.gaussian_generator = MultivariateNormal(torch.zeros(d_model), torch.eye(d_model))

        self.smoothL1 = torch.nn.SmoothL1Loss()
        self.mseloss = nn.MSELoss()
        self.klloss = KLDLossNoReduction()
        self._reset_parameters()
        self.d_model = d_model
        self.nhead = nhead

        self.gmm = GMM_head(d_model, greedy = True).to('cuda')
        self.gmm_loss = Log_Pdf(reduction='mean', pretrain = True, lambda_xy = 1., lambda_wh = 1., rel_gt = False, raw_batch_size=args.batch_size, KD_ON=True, Topk=-1)


    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def netE(self, X, embs, cond_token):

        bs, N, _ = X.shape
        x = self._4tod_fc(X)
        x = x + embs

        cls_token = self.netEemb.repeat(args.batch_size, 1, 1)
        x = torch.cat([cls_token, x], 1)
        pos = self.PositionalEncoder(x)
        x = x.permute(1, 0, 2); pos = pos.permute(1, 0, 2)
        rel_pos = BoxRelationalEmbedding(X)
        x = self.trencoder_1(x, pos = pos, rel_pos = rel_pos)

        output_token = x[0]
        output_token = torch.cat([output_token, cond_token], -1)

        mu1 = self.linear_mu1(output_token)
        logvar1 = self.linear_logvar1(output_token)    

        std1 = F.softplus(logvar1)

        dist1 = torch.distributions.Normal(mu1, std1)
        dist1_fix = torch.distributions.Normal(mu1.detach(), std1.detach())
        zs1 = dist1.rsample()


        return dist1, dist1_fix, zs1

    def netF(self, intial_xy):

        st = self.intial_xy_layer(intial_xy.float())

        cls_token = self.netFemb.repeat(args.batch_size, 1, 1)

        st = torch.cat([cls_token, st], 1)
        pos = self.PositionalEncoder(st)
        st = st.permute(1, 0, 2); pos = pos.permute(1, 0, 2)
        st = self.trencoder_2(st, pos = pos)

        cond_token = st[0]

        mu2 = self.linear_mu2(cond_token)
        logvar2 = self.linear_logvar2(cond_token)

        std2 = F.softplus(logvar2)

        dist2 = torch.distributions.Normal(mu2, std2)
        dist2_fix = torch.distributions.Normal(mu2.detach(), std2.detach())

        zs2 = dist2.rsample()

        return cond_token, st, zs2, dist2, dist2_fix

    def netZ(self, zs, mu):

        z_cond = torch.cat([zs, mu], 1)
        z_cond = self.z_layer(z_cond)

        return z_cond

    def netD(self, st, z_cond, embs):

        #st = self.intial_xy_layer_dec(intial_xy.float())

        tgt = z_cond.repeat(embs.shape[1], 1, 1).permute(1, 0, 2)
        pos = self.PositionalEncoder(tgt)
        pos = pos.permute(1, 0, 2)
        tgt = tgt.permute(1, 0, 2)
        embs = embs.permute(1, 0, 2)

        X_caps_1 = self.trdecoder_1(tgt, st, query_pos = embs)[-1]
        X_caps_2 = self.trdecoder_2(X_caps_1, st, query_pos = embs)[-1]
        X_caps_1 = X_caps_1.permute(1, 0, 2)
        X_caps_2 = X_caps_2.permute(1, 0, 2)
        coarse_wh, coarse_xy,  coarse_wh_gmm, coarse_xy_gmm, xy_pdf_score = self.gmm(X_caps_1, X_caps_2)
        coarse_gmm = torch.cat((coarse_xy_gmm, coarse_wh_gmm), dim=-1)
        coarse_box = torch.cat((coarse_xy, coarse_wh), dim=-1)
        coarse_gmm = coarse_gmm.reshape(coarse_gmm.size(0) * coarse_gmm.size(1), coarse_gmm.size(2))   

        return coarse_box, coarse_gmm


    def forward(self, X, Y, intial_xy, eta_step):

        embs = self.part_emb[Y.long()]
    
        ### ENCODER II

        cond_token, st, zs2, dist2, dist2_fix = self.netF(intial_xy)

        ### ENCODER I

        dist1, dist1_fix, zs1 = self.netE(X, embs, cond_token)
             
        ### Z_merge
        
        z_cond = self.netZ(zs1, dist2.mean)
    
        ### Decoder

        coarse_box, coarse_gmm = self.netD(st, zs1, embs)

        ### loss

        coarse_box_label = X.reshape(X.size(0) * X.size(1), X.size(2))
        L_recons, L_KLD = self.gmm_loss(coarse_gmm, coarse_box_label, False, Y) 

        distN = torch.distributions.Normal(torch.zeros_like(dist1.mean), torch.ones_like(dist1.stddev))

        L_kl = torch.distributions.kl_divergence(dist1, dist2).mean() 
        #L_kl = mmd_loss(dist1.rsample(), dist2.rsample()).mean() 

        loss = L_recons + args.beta*1*(L_kl) 

        return coarse_box, zs1, loss, L_recons, L_kl, L_KLD


    def validate(self, Y, intial_xy):
        
        with torch.no_grad():

            embs = self.part_emb[Y.long()]

            zs = torch.randn(args.batch_size, self.d_model).to('cuda:0')

            #zs = self.gaussian_generator.sample([BATCH_SIZE]).to('cuda:0')

            cond_token, st, zs2, dist2, dist2_fix = self.netF(intial_xy)
            #z_cond = self.netZ(zs, dist2.mean)
            coarse_box, coarse_gmm = self.netD(st, zs2, embs)

            return coarse_box


def main():

    wandb.init(settings=wandb.Settings(start_method='fork'), project="doodleformer", name = args.exp_name)
    
    data  = Dataset_JSON()

    dataloader = torch.utils.data.DataLoader(
        data, batch_size=args.batch_size, collate_fn=collate_fn, 
        drop_last=True, shuffle=True, num_workers=4)

    test_dataloader = torch.utils.data.DataLoader(
        data, batch_size=args.batch_size, collate_fn=collate_fn, 
        drop_last=True, shuffle=True, num_workers=4)


    model = Generator(normalize_before = True).to('cuda')

    # netG = nn.DataParallel(context_aware_generator(num_classes=10, output_dim=3).to('cuda:0'))
    # netG.load_state_dict(torch.load('../../models/layout2sketch-gpu4-birds-epoch-1000-bs32-context-class-graph-bboxadj/G_500.pth'))

    opt = torch.optim.Adam(model.parameters(), lr = args.learning_rate, betas=(0., 0.99))

    model.train()
    # netG.eval()

    def calculate_scores(epoch, test_dataloader):

        output_path = '../../output/'
    
        total_path = os.path.join(output_path, args.exp_name + str('-') + str(epoch))

        if os.path.isdir(total_path):

            shutil.rmtree(total_path)

        Path(output_path).mkdir(parents=True, exist_ok=True)
        Path(total_path).mkdir(parents=True, exist_ok=True)

        print ('calculating scores : epoch '+str(epoch))
        
        for idx, batch in enumerate(tqdm(test_dataloader)):

            real_images, label, bbox, intial_xy, raster_initial = batch 
            bbox = bbox.to('cuda')
            label = label.to('cuda')
            intial_xy = intial_xy.to('cuda')

            bbox_val = model.validate(label, intial_xy)

            bbox_val, bbox_val = fix_bboxs(bbox, bbox_val, bbox_val)
            z = torch.randn(real_images.size(0), 9, 128).to('cuda:0')
            fake_images = netG(z, bbox_val, y=label.long().to('cuda:0'))[0].detach()

            for i in range(fake_images.shape[0]):

                im = (1 - (fake_images[i].permute(1,2,0).cpu().numpy() + 1)/2)*255

                im =  cv2.cvtColor(cv2.resize(im, (64, 64)),  cv2.COLOR_BGR2GRAY)

                cv2.imwrite(os.path.join(total_path, 'image'+str(idx*args.batch_size + i)+'.jpg'), im)

            if len(os.listdir(total_path))>10000:

                break

        fid_value, d1, d2, CS1, CS2, SDS1, SDS2 = calculate_scores_given_paths(['../../data/bird_short_full_nodetail_64',total_path], 50, 1, 2048, 'birds', src_pkl_file='birds.pickle')

        return fid_value, d1, d2, CS1, CS2, SDS1, SDS2

    steps = 0

    for epoch in range(args.num_epochs):
        
        for idx, batch in enumerate(tqdm(dataloader)):
            
            opt.zero_grad()

            real_images, label, bbox, intial_xy, raster_initial = batch 
            bbox = bbox.to('cuda')
            label = label.to('cuda')
            intial_xy = intial_xy.to('cuda')

            eta_step = 1-(1-0.001)*(0.9995**steps)
   
            try:
                X_, zs, loss, L_recons, L_kl, L_KLD = model(bbox, label, intial_xy, eta_step)
            
                bbox_recons = X_.view(args.batch_size, N_PARTS, -1)

            except:
                continue

            steps = steps + 1
        

            loss.backward()
            opt.step()

            wandb.log({'loss': loss, 'L_recons' : L_recons, 'L_kl' : L_kl, 'L_KLD' : L_KLD, 'eta_step' : eta_step}) 

            if idx%args.vis_per_step==0:

                

                bbox_val = model.validate(label, intial_xy)
                bbox_val_2 = model.validate(label, intial_xy)

                z = torch.randn(real_images.size(0), 9, 128).to('cuda')

                bbox_val_2, bbox_val = fix_bboxs(bbox, bbox_val_2, bbox_val)


                imgs = []

                for j in range(8):

                    img1  = visalize_bboxs(dataset =  args.dataset, inp = np.ones((128, 128, 3))*255,
                                label = label[j].int().detach().cpu().numpy(), 
                                bbox = bbox[j].detach().cpu().numpy())

                    img2 = visalize_bboxs(dataset =  args.dataset, inp = np.ones((128, 128, 3))*255,
                                label = label[j].detach().cpu().numpy(), 
                                bbox = bbox_recons[j].detach().cpu().numpy())

                    img3 = visalize_bboxs(dataset =  args.dataset, inp = np.ones((128, 128, 3))*255,
                                label = label[j].detach().cpu().numpy(), 
                                bbox = bbox_val[j].detach().cpu().numpy())

                    img4 = visalize_bboxs(dataset =  args.dataset, inp = np.ones((128, 128, 3))*255,
                                label = label[j].detach().cpu().numpy(), 
                                bbox = bbox_val_2[j].detach().cpu().numpy())            

                    img = np.concatenate([img1, img2, img3, img4], 1)

                    imgs.append(img)

                wandb.log({
                        "bbox":[wandb.Image(im) for im in imgs]
                })

        print ({'epoch':epoch, 'loss': loss, 'L_recons' : L_recons, 'L_kl' : L_kl, 'L_KLD' : L_KLD, 'eta_step' : eta_step})

        if (epoch + 1)%args.save_per_epoch==0:

            torch.save(model.state_dict(), os.path.join(args.model_dir, args.exp_name, 'model%d.pth' % (epoch + 1)))


if __name__ == "__main__":
    

    parser = argparse.ArgumentParser(description='doodleformer-plnet-training-stage-1')

    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--dataset', type=str, default='sketch-bird')
    parser.add_argument('--data_path', type=str, default='../../data/doodledata.npy')
    parser.add_argument('--beta', type=int, default=1)
    parser.add_argument('--exp_name', type=str, default='ztolayout')
    parser.add_argument('--wandb_dir', type=str, default='.')
    parser.add_argument('--model_dir', type=str, default='../../models/')
    parser.add_argument('--save_per_epoch', type=int, default=20)
    parser.add_argument('--vis_per_step', type=int, default=200)
    parser.add_argument('--num_epochs', type=int, default=100000)
    parser.add_argument('--learning_rate', type=int, default=0.0001)

    args = parser.parse_args()

    if 'bird' in args.dataset:
        target_parts = ['eye', 'head', 'body', 'beak', 'legs', 'wings', 'mouth', 'tail', 'none']
        id_to_part = {0:'initial', 1:'eye', 4:'head', 3:'body', 2:'beak', 5:'legs', 8:'wings', 6:'mouth', 7:'tail', 9: 'none'}
        color_table = {'initial':np.array([45, 169, 145])/255., 'eye':np.array([0, 255, 0])/255., 'none':np.array([149, 165, 166])/255., 
                        'beak':np.array([176, 48, 96])/255., 'body':np.array([0, 0, 139])/255., 'details':np.array([171, 190, 191])/255.,
                        'head':np.array([255, 69, 0])/255., 'legs':np.array([255, 215, 0])/255., 'mouth':np.array([0, 255, 255])/255., 
                        'tail':np.array([255, 0, 255])/255., 'wings':np.array([100, 149, 237])/255.}
    elif 'creature' in  args.dataset or 'generic' in  args.dataset :# or 'fin' in folder or 'horn' in folder:
        target_parts = ['eye', 'arms', 'beak', 'mouth', 'body', 'ears', 'feet', 'fin', 
                        'hair', 'hands', 'head', 'horns', 'legs', 'nose', 'paws', 'tail', 'wings', 'none']
        id_to_part = { 0:'initial',  1:'eye',  2:'arms',  3:'beak',  4:'mouth',  5:'body',  6:'ears',  7:'feet',  8:'fin', 
                        9:'hair',  10:'hands',  11:'head',  12:'horns',  13:'legs',  14:'nose',  15:'paws',  16:'tail', 17:'wings', 18: 'none'}
        color_table = {'initial':np.array([45, 169, 145])/255., 'eye':np.array([243, 156, 18])/255., 'none':np.array([149, 165, 166])/255., 
                        'arms':np.array([211, 84, 0])/255., 'beak':np.array([41, 128, 185])/255., 'mouth':np.array([54, 153, 219])/255.,
                        'body':np.array([192, 57, 43])/255., 'ears':np.array([142, 68, 173])/255., 'feet':np.array([39, 174, 96])/255., 
                        'fin':np.array([69, 85, 101])/255., 'hair':np.array([127, 140, 141])/255., 'hands':np.array([45, 63, 81])/255.,
                        'head':np.array([241, 197, 17])/255., 'horns':np.array([51, 205, 117])/255., 'legs':np.array([232, 135, 50])/255., 
                        'nose':np.array([233, 90, 75])/255., 'paws':np.array([160, 98, 186])/255., 'tail':np.array([58, 78, 99])/255., 
                        'wings':np.array([198, 203, 207])/255., 'details':np.array([171, 190, 191])/255.}

    N_PARTS = len(target_parts)
    
    os.environ['WANDB_DIR'] = args.wandb_dir
    Path(args.model_dir).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(args.model_dir, args.exp_name)).mkdir(parents=True, exist_ok=True)

    main()
        

