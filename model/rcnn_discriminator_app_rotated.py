import torch
import torch.nn as nn
import torch.nn.functional as F
#from .roi_layers import ROIAlign, ROIPool
from torchvision.ops import RoIAlign, RoIPool
from detectron2.layers.roi_align_rotated import ROIAlignRotated
from utils.util import *
from utils.bilinear import *
from .norm_module import *
from .mask_regression import *
import copy
import scipy.sparse as sp
from torch.nn.modules.module import Module
from itertools import permutations
import math
#from torch_scatter import scatter


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


def conv2d(in_feat, out_feat, kernel_size=3, stride=1, pad=1, spectral_norm=True):
    conv = nn.Conv2d(in_feat, out_feat, kernel_size, stride, pad)
    if spectral_norm:
        return nn.utils.spectral_norm(conv, eps=1e-4)
    else:
        return conv


class GatedPooling(nn.Module):
    '''
    Modified Version of Global Pooling Layer from the “Gated Graph Sequence Neural Networks” paper
    Parameters:
    ----------
        node_dim: Dimension of node features
        edge_dim: Dimension of edge features
    '''

    def __init__(self, node_dim, pooling_dim):
        super(GatedPooling, self).__init__()

        ###############################################################
        # Gates to compute attention scores
        self.hgate_node = nn.Sequential(
            nn.Linear(node_dim, 1)
        )

        ##############################################################
        #Layers to tranfrom features before combinig
        self.htheta_node = nn.Sequential(
            nn.Linear(node_dim, node_dim),
            nn.ReLU()
        )

        ################################################################
        #Final pooling layer
        self.poolingLayer = nn.Sequential(
            nn.Linear(node_dim, pooling_dim)
        )
    def forward(self, node_features, batch_list):

        node_alpha = self.hgate_node(node_features)
        node_pool = scatter(node_alpha*node_features, batch_list, dim=0, reduce="sum")

        return self.poolingLayer(node_pool)

def conv2d(in_feat, out_feat, kernel_size=3, stride=1, pad=1, spectral_norm=True):
    conv = nn.Conv2d(in_feat, out_feat, kernel_size, stride, pad)
    if spectral_norm:
        return nn.utils.spectral_norm(conv, eps=1e-4)
    else:
        return conv

class ResnetEncoder128(nn.Module):
    def __init__(self, input_dim=3, ch=64):
        super(ResnetEncoder128, self).__init__()
      

        self.block1 = OptimizedBlock(input_dim, ch, downsample=True)
        self.block2 = ResBlock(ch, ch * 2, downsample=True)
        self.block3 = ResBlock(ch * 2, ch * 4, downsample=True)
        self.block4 = ResBlock(ch * 4, ch * 8, downsample=True)
        self.block5 = ResBlock(ch * 8, ch * 16, downsample=True)
        #self.block6 = ResBlock(ch * 16, ch * 16, downsample=False)

    def forward(self, x):

   
        x = self.block1(x)
        # 64x64
        x1 = self.block2(x)
        # 32x32
        x2 = self.block3(x1)
        # 16x16
        x = self.block4(x2)
        # 8x8
        x = self.block5(x)
        # 4x4
        return x
        
class ResnetDiscriminator128(nn.Module):
    def __init__(self, num_classes=0, input_dim=3, ch=64):
        super(ResnetDiscriminator128, self).__init__()
        self.num_classes = num_classes

        self.block1 = OptimizedBlock(input_dim, ch, downsample=True)
        self.block2 = ResBlock(ch, ch * 2, downsample=True)
        self.block3 = ResBlock(ch * 2, ch * 4, downsample=True)
        self.block4 = ResBlock(ch * 4, ch * 8, downsample=True)
        self.block5 = ResBlock(ch * 8, ch * 16, downsample=True)
        self.block6 = ResBlock(ch * 16, ch * 16, downsample=False)
        self.l7 = nn.utils.spectral_norm(nn.Linear(ch * 16, 1))
        self.activation = nn.ReLU()

        self.roi_align_s = RoIAlign((8, 8), 1.0 / 4.0, int(0))
        self.roi_align_l = RoIAlign((8, 8), 1.0 / 8.0, int(0))

        self.block_obj3 = ResBlock(ch * 2, ch * 4, downsample=False)
        self.block_obj4 = ResBlock(ch * 4, ch * 8, downsample=False)
        self.block_obj5 = ResBlock(ch * 8, ch * 16, downsample=True)
        self.l_obj = nn.utils.spectral_norm(nn.Linear(ch * 16, 1))
        self.l_y = nn.utils.spectral_norm(nn.Embedding(num_classes, ch * 16))

    def forward(self, x, y=None, bbox=None):
        b = bbox.size(0)
        # 128x128
        x = self.block1(x)
        # 64x64
        x1 = self.block2(x)
        # 32x32
        x2 = self.block3(x1)
        # 16x16
        x = self.block4(x2)
        # 8x8
        x = self.block5(x)
        # 4x4
        x = self.block6(x)
        x = self.activation(x)
        x = torch.sum(x, dim=(2, 3))
        out_im = self.l7(x)

        # obj path
        # seperate different path
        s_idx = ((bbox[:, 3] - bbox[:, 1]) < 64) * ((bbox[:, 4] - bbox[:, 2]) < 64)
        bbox_l, bbox_s = bbox[~s_idx], bbox[s_idx]
        y_l, y_s = y[~s_idx], y[s_idx]

        obj_feat_s = self.block_obj3(x1)
        obj_feat_s = self.block_obj4(obj_feat_s)
        # print(obj_feat_s.shape)
        obj_feat_s = self.roi_align_s(obj_feat_s, bbox_s)

        obj_feat_l = self.block_obj4(x2)
        obj_feat_l = self.roi_align_l(obj_feat_l, bbox_l)

        obj_feat = torch.cat([obj_feat_l, obj_feat_s], dim=0)
        y = torch.cat([y_l, y_s], dim=0)
        obj_feat = self.block_obj5(obj_feat)
        obj_feat = self.activation(obj_feat)
        obj_feat = torch.sum(obj_feat, dim=(2, 3))
        out_obj = self.l_obj(obj_feat)
        out_obj = out_obj + torch.sum(self.l_y(y).view(b, -1) * obj_feat.view(b, -1), dim=1, keepdim=True)

        return out_im, out_obj


class ResnetDiscriminator128_app(nn.Module):
    def __init__(self, num_classes=0, input_dim=3, ch=64):
        super(ResnetDiscriminator128_app, self).__init__()
        self.num_classes = num_classes

        self.block1 = OptimizedBlock(input_dim, ch, downsample=True)
        self.block2 = ResBlock(ch, ch * 2, downsample=True)
        self.block3 = ResBlock(ch * 2, ch * 4, downsample=True)
        self.block4 = ResBlock(ch * 4, ch * 8, downsample=True)
        self.block5 = ResBlock(ch * 8, ch * 16, downsample=True)
        self.block6 = ResBlock(ch * 16, ch * 16, downsample=False)
        self.l7 = nn.utils.spectral_norm(nn.Linear(ch * 16, 1))
        self.activation = nn.ReLU()

        self.obj_graph = GraphFunc(1024)
        #self.g_pool = GatedPooling(1024, 1024)
        self.roi_align_s = ROIAlignRotated((8, 8), 1.0 / 4.0, int(0))
        self.roi_align_l = ROIAlignRotated((8, 8), 1.0 / 8.0, int(0))

        self.block_obj3 = ResBlock(ch * 2, ch * 4, downsample=False)
        self.block_obj4 = ResBlock(ch * 4, ch * 8, downsample=False)
        self.block_obj5 = ResBlock(ch * 8, ch * 16, downsample=True)
        self.l_obj = nn.utils.spectral_norm(nn.Linear(ch * 16, 1))

        self.l_graph_obj = nn.utils.spectral_norm(nn.Linear(1024, 9+1))
        self.l_y = nn.utils.spectral_norm(nn.Embedding(num_classes, ch * 16))
        # apperance discriminator
        self.app_conv = ResBlock(ch * 8, ch * 8, downsample=False)
        self.l_y_app = nn.utils.spectral_norm(nn.Embedding(num_classes, ch * 8))
        self.app = nn.utils.spectral_norm(nn.Linear(ch * 16, 1))
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()


    def forward(self, x, y=None, bbox=None, idx = None):
        b = bbox.size(0)
        # 128x128
        x = self.block1(x)
        # 64x64
        x1 = self.block2(x)
        # 32x32
        x2 = self.block3(x1)
        # 16x16
        x = self.block4(x2)
        # 8x8
        x = self.block5(x)
        # 4x4
        x = self.block6(x)
        x = self.activation(x)
        x = torch.sum(x, dim=(2, 3))
        out_im = self.l7(x)

        # obj path
        # seperate different path
        s_idx = ((bbox[:, 3]) < 64) * ((bbox[:, 4]) < 64)
        bbox_l, bbox_s = bbox[~s_idx], bbox[s_idx]

        idx_ = torch.cat([idx[torch.where(~s_idx)[0]],idx[torch.where(s_idx)[0]]], -1)
        
        y_l, y_s = y[~s_idx], y[s_idx]

        obj_feat_s = self.block_obj3(x1)
        obj_feat_s = self.block_obj4(obj_feat_s)

        obj_feat_s = self.roi_align_s(obj_feat_s, bbox_s)

        obj_feat_l = self.block_obj4(x2)

        obj_feat_l = self.roi_align_l(obj_feat_l, bbox_l)

        obj_feat = torch.cat([obj_feat_l, obj_feat_s], dim=0)
        y = torch.cat([y_l, y_s], dim=0)
        # apperance
        app_feat = self.app_conv(obj_feat) 
        app_feat = self.activation(app_feat)

        s1, s2, s3, s4 = app_feat.size()
        app_feat = app_feat.view(s1, s2, s3 * s4)
        app_gram = torch.bmm(app_feat, app_feat.permute(0, 2, 1)) / s2

        app_y = self.l_y_app(y).unsqueeze(1).expand(s1, s2, s2)
        app_all = torch.cat([app_gram, app_y], dim=-1)
        out_app = self.app(app_all).sum(1) / s2

        # original one for single instance
        obj_feat = self.block_obj5(obj_feat)
        obj_feat = self.activation(obj_feat)
        obj_feat = torch.sum(obj_feat, dim=(2, 3))
        # print(obj_feat.shape)
        out_obj = self.l_obj(obj_feat)

        out_obj = out_obj + torch.sum(self.l_y(y).view(b, -1) * obj_feat.view(b, -1), dim=1, keepdim=True)

        obj_feat_1 = obj_feat[torch.argsort(idx_)]
        y_1 = y[torch.argsort(idx_)]

        idx_1 = torch.sort(idx_)[0]

        obj_feat_zeros = torch.zeros((32*9, 1024)).to(obj_feat.device)
        y_zeros = torch.zeros((32*9)).to(y.device)

        for i,j,k in zip(obj_feat_1.unbind(0), y_1.unbind(0), idx_1.unbind(0)):

            obj_feat_zeros[k] = obj_feat_zeros[k] + i

            y_zeros[k] = y_zeros[k] + j

        obj_feat_2 = obj_feat_zeros.view(32, 9, -1)

        y_2 = y_zeros.view(32, 9)

        obj_graph_feat = self.obj_graph(obj_feat_2 , y_2).view(-1, 1024)

        y_2 = y_2.view(-1)

        obj_graph_feat_1 = self.l_graph_obj(obj_graph_feat)

        loss_class = self.cross_entropy_loss(obj_graph_feat_1, y_2.view(-1).long())
        
        #out_obj = self.l_obj(obj_graph_feat[idx_])
        #out_obj = out_obj + torch.sum(self.l_y(y).view(b, -1) * obj_feat.view(b, -1), dim=1, keepdim=True)

        #g_pool_feat = self.g_pool(obj_graph_feat[idx_], y_2.long()[idx_])
        #out_obj = self.l_obj(g_pool_feat)

        return out_im, out_obj, out_app, loss_class



class ResnetDiscriminator128_graph(nn.Module):
    def __init__(self, num_classes=0, input_dim=3, ch=64):
        super(ResnetDiscriminator128_graph, self).__init__()
        self.num_classes = num_classes

        self.block1 = OptimizedBlock(input_dim, ch, downsample=True)
        self.block2 = ResBlock(ch, ch * 2, downsample=True)
        self.block3 = ResBlock(ch * 2, ch * 4, downsample=True)
        self.block4 = ResBlock(ch * 4, ch * 8, downsample=True)
        self.block5 = ResBlock(ch * 8, ch * 16, downsample=True)
        self.block6 = ResBlock(ch * 16, ch * 16, downsample=False)
        self.l7 = nn.utils.spectral_norm(nn.Linear(ch * 16, 1))
        self.activation = nn.ReLU()

        self.obj_graph = GraphFunc(1024)
        
        self.g_pool = GatedPooling(1024, 512)
        self.f_pool = GatedPooling(308, 512)

        self.roi_align_s = ROIAlignRotated((8, 8), 1.0 / 4.0, int(0))
        self.roi_align_l = ROIAlignRotated((8, 8), 1.0 / 8.0, int(0))

        self.block_obj3 = ResBlock(ch * 2, ch * 4, downsample=False)
        self.block_obj4 = ResBlock(ch * 4, ch * 8, downsample=False)
        self.block_obj5 = ResBlock(ch * 8, ch * 16, downsample=True)
        self.l_obj = nn.utils.spectral_norm(nn.Linear(ch * 16, 1))

        self.l_graph_obj = nn.utils.spectral_norm(nn.Linear(1024, 9+1))
        self.l_y = nn.utils.spectral_norm(nn.Embedding(num_classes, ch * 16))
        # apperance discriminator
        self.app_conv = ResBlock(ch * 8, ch * 8, downsample=False)
        self.l_y_app = nn.utils.spectral_norm(nn.Embedding(num_classes, ch * 8))
        self.app = nn.utils.spectral_norm(nn.Linear(ch * 16, 1))
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()


    def forward(self, x, y=None, bbox=None, idx = None, f_graph = None):
        b = bbox.size(0)
        # 128x128
        x = self.block1(x)
        # 64x64
        x1 = self.block2(x)
        # 32x32
        x2 = self.block3(x1)
        # 16x16
        x = self.block4(x2)
        # 8x8
        x = self.block5(x)
        # 4x4
        x = self.block6(x)
        x = self.activation(x)
        x = torch.sum(x, dim=(2, 3))
        out_im = self.l7(x)

        # obj path
        # seperate different path
        s_idx = ((bbox[:, 3]) < 64) * ((bbox[:, 4]) < 64)
        bbox_l, bbox_s = bbox[~s_idx], bbox[s_idx]

        idx_ = torch.cat([idx[torch.where(~s_idx)[0]],idx[torch.where(s_idx)[0]]], -1)
        
        y_l, y_s = y[~s_idx], y[s_idx]

        obj_feat_s = self.block_obj3(x1)
        obj_feat_s = self.block_obj4(obj_feat_s)

        obj_feat_s = self.roi_align_s(obj_feat_s, bbox_s)

        obj_feat_l = self.block_obj4(x2)

        obj_feat_l = self.roi_align_l(obj_feat_l, bbox_l)

        obj_feat = torch.cat([obj_feat_l, obj_feat_s], dim=0)
        y = torch.cat([y_l, y_s], dim=0)

        obj_feat = self.block_obj5(obj_feat)
        obj_feat = self.activation(obj_feat)
        obj_feat = torch.sum(obj_feat, dim=(2, 3))
        # print(obj_feat.shape)

        #out_obj = self.l_obj(obj_feat)

        #out_obj = out_obj + torch.sum(self.l_y(y).view(b, -1) * obj_feat.view(b, -1), dim=1, keepdim=True)

        obj_feat_1 = obj_feat[torch.argsort(idx_)]
        y_1 = y[torch.argsort(idx_)]

        idx_1 = torch.sort(idx_)[0]

        obj_feat_zeros = torch.zeros((32*9, 1024)).to(obj_feat.device)
        y_zeros = torch.zeros((32*9)).to(y.device)

        for i,j,k in zip(obj_feat_1.unbind(0), y_1.unbind(0), idx_1.unbind(0)):

            obj_feat_zeros[k] = obj_feat_zeros[k] + i

            y_zeros[k] = y_zeros[k] + j

        obj_feat_2 = obj_feat_zeros.view(32, 9, -1)

        y_2 = y_zeros.view(32, 9)

        obj_graph_feat = self.obj_graph(obj_feat_2 , y_2).view(-1, 1024)

        y_2 = y_2.view(-1)

        obj_graph_feat_1 = self.l_graph_obj(obj_graph_feat)

        loss_class = self.cross_entropy_loss(obj_graph_feat_1, y_2.view(-1).long())

        g_pool_feat = self.g_pool(obj_graph_feat[idx_], y_2.long()[idx_])
        f_pool_feat = self.f_pool(f_graph[idx_], y_2.long()[idx_])
        
        pool_feat  = torch.cat([g_pool_feat, f_pool_feat], 1)

        out_obj = self.l_obj(pool_feat)

        return out_im, out_obj, loss_class



class ResnetDiscriminator64(nn.Module):
    def __init__(self, num_classes=0, input_dim=3, ch=64):
        super(ResnetDiscriminator64, self).__init__()
        self.num_classes = num_classes

        self.block1 = OptimizedBlock(input_dim, ch, downsample=False)
        self.block2 = ResBlock(ch, ch * 2, downsample=False)
        self.block3 = ResBlock(ch * 2, ch * 4, downsample=True)
        self.block4 = ResBlock(ch * 4, ch * 8, downsample=True)
        self.block5 = ResBlock(ch * 8, ch * 16, downsample=True)
        self.l_im = nn.utils.spectral_norm(nn.Linear(ch * 16, 1))
        self.activation = nn.ReLU()

        # object path
        self.roi_align = ROIAlign((8, 8), 1.0 / 2.0, 0)
        self.block_obj4 = ResBlock(ch * 4, ch * 8, downsample=True)
        self.l_obj = nn.utils.spectral_norm(nn.Linear(ch * 8, 1))
        self.l_y = nn.utils.spectral_norm(nn.Embedding(num_classes, ch * 8))

        self.init_parameter()

    def forward(self, x, y=None, bbox=None):
        b = bbox.size(0)
        # 64x64
        x = self.block1(x)
        # 64x64
        x = self.block2(x)
        # 32x32
        x1 = self.block3(x)
        # 16x16
        x = self.block4(x1)
        # 8x8
        x = self.block5(x)
        x = self.activation(x)
        x = torch.mean(x, dim=(2, 3))
        out_im = self.l_im(x)

        # obj path
        obj_feat = self.roi_align(x1, bbox)
        obj_feat = self.block_obj4(obj_feat)
        obj_feat = self.activation(obj_feat)
        obj_feat = torch.sum(obj_feat, dim=(2, 3))
        out_obj = self.l_obj(obj_feat)
        out_obj = out_obj + torch.sum(self.l_y(y).view(b, -1) * obj_feat.view(b, -1), dim=1, keepdim=True)

        return out_im, out_obj

    def init_parameter(self):
        for k in self.named_parameters():
            if k[1].dim() > 1:
                torch.nn.init.orthogonal_(k[1])
            if k[0][-4:] == 'bias':
                torch.nn.init.constant_(k[1], 0)


class ResnetDiscriminator256(nn.Module):
    def __init__(self, num_classes=0, input_dim=3, ch=64):
        super(ResnetDiscriminator256, self).__init__()
        self.num_classes = num_classes

        self.block1 = OptimizedBlock(input_dim, ch, downsample=True)
        self.block2 = ResBlock(ch, ch * 2, downsample=True)
        self.block3 = ResBlock(ch * 2, ch * 4, downsample=True)
        self.block4 = ResBlock(ch * 4, ch * 8, downsample=True)
        self.block5 = ResBlock(ch * 8, ch * 8, downsample=True)
        self.block6 = ResBlock(ch * 8, ch * 16, downsample=True)
        self.block7 = ResBlock(ch * 16, ch * 16, downsample=False)
        self.l8 = nn.utils.spectral_norm(nn.Linear(ch * 16, 1))
        self.activation = nn.ReLU()

        self.roi_align_s = ROIAlign((8, 8), 1.0 / 8.0, int(0))
        self.roi_align_l = ROIAlign((8, 8), 1.0 / 16.0, int(0))

        self.block_obj4 = ResBlock(ch * 4, ch * 8, downsample=False)
        self.block_obj5 = ResBlock(ch * 8, ch * 8, downsample=False)
        self.block_obj6 = ResBlock(ch * 8, ch * 16, downsample=True)
        self.l_obj = nn.utils.spectral_norm(nn.Linear(ch * 16, 1))
        self.l_y = nn.utils.spectral_norm(nn.Embedding(num_classes, ch * 16))

    def forward(self, x, y=None, bbox=None):
        b = bbox.size(0)
        # 256x256
        x = self.block1(x)
        # 128x128
        x = self.block2(x)
        # 64x64
        x1 = self.block3(x)
        # 32x32
        x2 = self.block4(x1)
        # 16x16
        x = self.block5(x2)
        # 8x8
        x = self.block6(x)
        # 4x4
        x = self.block7(x)
        x = self.activation(x)
        x = torch.sum(x, dim=(2, 3))
        out_im = self.l8(x)

        # obj path
        # seperate different path
        s_idx = ((bbox[:, 3] - bbox[:, 1]) < 128) * ((bbox[:, 4] - bbox[:, 2]) < 128)
        bbox_l, bbox_s = bbox[~s_idx], bbox[s_idx]
        y_l, y_s = y[~s_idx], y[s_idx]

        obj_feat_s = self.block_obj4(x1)
        obj_feat_s = self.block_obj5(obj_feat_s)
        obj_feat_s = self.roi_align_s(obj_feat_s, bbox_s)

        obj_feat_l = self.block_obj5(x2)
        obj_feat_l = self.roi_align_l(obj_feat_l, bbox_l)

        obj_feat = torch.cat([obj_feat_l, obj_feat_s], dim=0)
        y = torch.cat([y_l, y_s], dim=0)
        obj_feat = self.block_obj6(obj_feat)
        obj_feat = self.activation(obj_feat)
        obj_feat = torch.sum(obj_feat, dim=(2, 3))
        out_obj = self.l_obj(obj_feat)
        out_obj = out_obj + torch.sum(self.l_y(y).view(b, -1) * obj_feat.view(b, -1), dim=1, keepdim=True)

        return out_im, out_obj

class OptimizedBlock(nn.Module):
    def __init__(self, in_ch, out_ch, ksize=3, pad=1, downsample=False):
        super(OptimizedBlock, self).__init__()
        self.conv1 = conv2d(in_ch, out_ch, ksize, 1, pad)
        self.conv2 = conv2d(out_ch, out_ch, ksize, 1, pad)
        self.c_sc = conv2d(in_ch, out_ch, 1, 1, 0)
        self.activation = nn.ReLU()
        self.downsample = downsample

    def forward(self, in_feat):
        x = in_feat
        x = self.activation(self.conv1(x))
        x = self.conv2(x)
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        return x + self.shortcut(in_feat)

    def shortcut(self, x):
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        return self.c_sc(x)


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, ksize=3, pad=1, downsample=False):
        super(ResBlock, self).__init__()
        self.conv1 = conv2d(in_ch, out_ch, ksize, 1, pad)
        self.conv2 = conv2d(out_ch, out_ch, ksize, 1, pad)
        self.activation = nn.ReLU()
        self.downsample = downsample
        self.learnable_sc = (in_ch != out_ch) or downsample
        if self.learnable_sc:
            self.c_sc = conv2d(in_ch, out_ch, 1, 1, 0)

    def residual(self, in_feat):
        x = in_feat
        x = self.conv1(self.activation(x))
        x = self.conv2(self.activation(x))
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        return x

    def shortcut(self, x):
        if self.learnable_sc:
            x = self.c_sc(x)
            if self.downsample:
                x = F.avg_pool2d(x, 2)
        return x

    def forward(self, in_feat):
        return self.residual(in_feat) + self.shortcut(in_feat)


class CombineDiscriminator256(nn.Module):
    def __init__(self, num_classes=81):
        super(CombineDiscriminator256, self).__init__()
        self.obD = ResnetDiscriminator256(num_classes=num_classes, input_dim=3)

    def forward(self, images, bbox, label, mask=None):
        idx = torch.arange(start=0, end=images.size(0),
                           device=images.device).view(images.size(0),
                                                      1, 1).expand(-1, bbox.size(1), -1).float()
        bbox[:, :, 2] = bbox[:, :, 2] + bbox[:, :, 0]
        bbox[:, :, 3] = bbox[:, :, 3] + bbox[:, :, 1]
        bbox = bbox * images.size(2)
        bbox = torch.cat((idx, bbox.float()), dim=2)
        bbox = bbox.view(-1, 5)
        label = label.view(-1)

        idx = (label != 0).nonzero().view(-1)
        bbox = bbox[idx]
        label = label[idx]
        d_out_img, d_out_obj = self.obD(images, label, bbox)
        return d_out_img, d_out_obj


class CombineDiscriminator128(nn.Module):
    def __init__(self, num_classes=81):
        super(CombineDiscriminator128, self).__init__()
        self.obD = ResnetDiscriminator128(num_classes=num_classes, input_dim=3)

    def forward(self, images, bbox, label, mask=None):
        idx = torch.arange(start=0, end=images.size(0),
                           device=images.device).view(images.size(0),
                                                      1, 1).expand(-1, bbox.size(1), -1).float()
        #idx = idx.cuda()
        # print(bbox)
        bbox = bbox.cuda()
        bbox[:, :, 2] = bbox[:, :, 2] + bbox[:, :, 0]
        bbox[:, :, 3] = bbox[:, :, 3] + bbox[:, :, 1]
        bbox = bbox * images.size(2)
        bbox = torch.cat((idx, bbox.float()), dim=2)
        bbox = bbox.view(-1, 5)
        label = label.view(-1)

        idx = (label != 0).nonzero().view(-1)
        bbox = bbox[idx]
        label = label[idx]
        d_out_img, d_out_obj = self.obD(images, label, bbox)
        return d_out_img, d_out_obj


class CombineDiscriminator128_app(nn.Module):
    def __init__(self, num_classes=81):
        super(CombineDiscriminator128_app, self).__init__()
        self.obD = ResnetDiscriminator128_app(num_classes=num_classes, input_dim=3)

    def forward(self, images, bbox, label, mask=None):
        idx = torch.arange(start=0, end=images.size(0),
                           device=images.device).view(images.size(0),
                                                      1, 1).expand(-1, bbox.size(1), -1).float()
        #idx = idx.cuda()
        # print(bbox)
        bbox = bbox.cuda()
        #bbox[:, :, 2] = bbox[:, :, 2] + bbox[:, :, 0]
        #bbox[:, :, 3] = bbox[:, :, 3] + bbox[:, :, 1]
        bbox[:, :, 0] = bbox[:, :, 0]*images.size(2)
        bbox[:, :, 1] = bbox[:, :, 1]*images.size(2)
        bbox[:, :, 2] = bbox[:, :, 2]*images.size(2)
        bbox[:, :, 3] = bbox[:, :, 3]*images.size(2)
        bbox[:, :, 4] = bbox[:, :, 4]*90

        #bbox = bbox * images.size(2)
        bbox = torch.cat((idx, bbox.float()), dim=2)
        bbox = bbox.view(-1, 6)
        label = label.view(-1)

        #idx = (label != 0).nonzero().view(-1)
        idx = torch.where(label != 0)[0]
        bbox = bbox[idx]
        label = label[idx]
        # print(bbox.shape)
        # print(label.shape)
        d_out_img, d_out_obj, out_app, loss_class = self.obD(images, label, bbox, idx)
        return d_out_img, d_out_obj, out_app, loss_class


class CombineDiscriminator128_graph(nn.Module):
    def __init__(self, num_classes=81):
        super(CombineDiscriminator128_graph, self).__init__()
        self.obD = ResnetDiscriminator128_graph(num_classes=num_classes, input_dim=3)

    def forward(self, images, bbox, label, mask=None, f_graph = None):
        idx = torch.arange(start=0, end=images.size(0),
                           device=images.device).view(images.size(0),
                                                      1, 1).expand(-1, bbox.size(1), -1).float()
        #idx = idx.cuda()
        # print(bbox)
        bbox = bbox.cuda()
        #bbox[:, :, 2] = bbox[:, :, 2] + bbox[:, :, 0]
        #bbox[:, :, 3] = bbox[:, :, 3] + bbox[:, :, 1]
        bbox[:, :, 0] = bbox[:, :, 0]*images.size(2)
        bbox[:, :, 1] = bbox[:, :, 1]*images.size(2)
        bbox[:, :, 2] = bbox[:, :, 2]*images.size(2)
        bbox[:, :, 3] = bbox[:, :, 3]*images.size(2)
        bbox[:, :, 4] = bbox[:, :, 4]*90

        #bbox = bbox * images.size(2)
        bbox = torch.cat((idx, bbox.float()), dim=2)
        bbox = bbox.view(-1, 6)
        label = label.view(-1)

        #idx = (label != 0).nonzero().view(-1)
        idx = torch.where(label != 0)[0]
        bbox = bbox[idx]
        label = label[idx]
        # print(bbox.shape)
        # print(label.shape)
        d_out_img, d_out_obj, loss_class = self.obD(images, label, bbox, idx, f_graph)
        return d_out_img, d_out_obj, loss_class


class CombineDiscriminator64(nn.Module):
    def __init__(self, num_classes=81):
        super(CombineDiscriminator64, self).__init__()
        self.obD = ResnetDiscriminator64(num_classes=num_classes, input_dim=3)

    def forward(self, images, bbox, label, mask=None):
        idx = torch.arange(start=0, end=images.size(0),
                           device=images.device).view(images.size(0),
                                                      1, 1).expand(-1, bbox.size(1), -1).float()
        bbox[:, :, 2] = bbox[:, :, 2] + bbox[:, :, 0]
        bbox[:, :, 3] = bbox[:, :, 3] + bbox[:, :, 1]
        bbox = bbox * images.size(2)
        bbox = torch.cat((idx, bbox.float()), dim=2)
        bbox = bbox.view(-1, 5)
        label = label.view(-1)

        idx = (label != 0).nonzero().view(-1)
        bbox = bbox[idx]
        label = label[idx]
        d_out_img, d_out_obj = self.obD(images, label, bbox)
        return d_out_img, d_out_obj
