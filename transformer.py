# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR Transformer class.
Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""

import copy
from typing import Optional, List
from torch.autograd import Variable
import time
import torch
import torch.nn.functional as F
from torch import nn, Tensor



def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx

def gmul(input):
    W, x = input
    # x is a tensor of size (bs, N, num_features)
    # W is a tensor of size (bs, N, N, J)
    x_size = x.size()
    W_size = W.size()
    N = W_size[-2]
    W = W.split(1, 3)
    W = torch.cat(W, 1).squeeze(3) # W is now a tensor of size (bs, J*N, N)
    output = torch.bmm(W, x) # output has size (bs, J*N, num_features)
    output = output.split(N, 1)
    output = torch.cat(output, 2) # output has size (bs, N, J*num_features)
    return output

class Gconv(nn.Module):
    def __init__(self, nf_input, nf_output, J, bn_bool=True):
        super(Gconv, self).__init__()
        self.J = J
        self.num_inputs = J*nf_input
        self.num_outputs = nf_output
        self.fc = nn.Linear(self.num_inputs, self.num_outputs)

        self.bn_bool = bn_bool
        if self.bn_bool:
            self.bn = nn.BatchNorm1d(self.num_outputs)

    def forward(self, input):
        W = input[0]
        x = gmul(input) # out has size (bs, N, num_inputs)
        #if self.J == 1:
        #    x = torch.abs(x)
        x_size = x.size()
        x = x.contiguous()
        x = x.view(-1, self.num_inputs)
        x = self.fc(x) # has size (bs*N, num_outputs)

        if self.bn_bool:
            x = self.bn(x)

        x = x.view(*x_size[:-1], self.num_outputs)
        return W, x


class Wcompute(nn.Module):
    def __init__(self, input_features, nf, operator='J2', activation='softmax', ratio=[2,2,1,1], num_operators=1, drop=False):
        super(Wcompute, self).__init__()
        self.num_features = nf
        self.operator = operator
        self.conv2d_1 = nn.Conv2d(input_features, int(nf * ratio[0]), 1, stride=1)
        self.bn_1 = nn.BatchNorm2d(int(nf * ratio[0]))
        self.drop = drop
        if self.drop:
            self.dropout = nn.Dropout(0.3)
        self.conv2d_2 = nn.Conv2d(int(nf * ratio[0]), int(nf * ratio[1]), 1, stride=1)
        self.bn_2 = nn.BatchNorm2d(int(nf * ratio[1]))
        self.conv2d_3 = nn.Conv2d(int(nf * ratio[1]), nf*ratio[2], 1, stride=1)
        self.bn_3 = nn.BatchNorm2d(nf*ratio[2])
        self.conv2d_4 = nn.Conv2d(nf*ratio[2], nf*ratio[3], 1, stride=1)
        self.bn_4 = nn.BatchNorm2d(nf*ratio[3])
        self.conv2d_last = nn.Conv2d(nf, num_operators, 1, stride=1)
        self.activation = activation

    def forward(self, x, W_id):
        W1 = x.unsqueeze(2)
        W2 = torch.transpose(W1, 1, 2) #size: bs x N x N x num_features
        W_new = torch.abs(W1 - W2) #size: bs x N x N x num_features
        W_new = torch.transpose(W_new, 1, 3) #size: bs x num_features x N x N

        W_new = self.conv2d_1(W_new)
        W_new = self.bn_1(W_new)
        W_new = F.leaky_relu(W_new)
        if self.drop:
            W_new = self.dropout(W_new)

        W_new = self.conv2d_2(W_new)
        W_new = self.bn_2(W_new)
        W_new = F.leaky_relu(W_new)

        W_new = self.conv2d_3(W_new)
        W_new = self.bn_3(W_new)
        W_new = F.leaky_relu(W_new)

        W_new = self.conv2d_4(W_new)
        W_new = self.bn_4(W_new)
        W_new = F.leaky_relu(W_new)

        W_new = self.conv2d_last(W_new)
        W_new = torch.transpose(W_new, 1, 3) #size: bs x N x N x 1

        if self.activation == 'softmax':
            W_new = W_new - W_id.expand_as(W_new) * 1e8
            
            W_new = torch.transpose(W_new, 2, 3)
            # Applying Softmax
            W_new = W_new.contiguous()
            W_new_size = W_new.size()
            W_new = W_new.view(-1, W_new.size(3))
            W_new = F.softmax(W_new)
            W_new = W_new.view(W_new_size)
            # Softmax applied
            W_new = torch.transpose(W_new, 2, 3)
            

        elif self.activation == 'sigmoid':
            W_new = F.sigmoid(W_new)
            W_new *= (1 - W_id)
        elif self.activation == 'none':
            W_new *= (1 - W_id)
        else:
            raise (NotImplementedError)
        
        return W_new



class GNN(nn.Module):
    def __init__(self, input_features, nhead):
        super(GNN, self).__init__()
        

        self.d_model = input_features
        self.nhead = nhead
        self.head_dim = self.d_model//self.nhead   

        self.input_features =  self.d_model//self.nhead   
        self.nf =  self.d_model//self.nhead    

        self.module_w = Wcompute(self.input_features, self.nf, operator='J2', activation='softmax', ratio=[2, 2, 1, 1])
        self.module_l = Gconv(self.input_features,self.input_features, 2)
        self.fc = nn.Linear(2*self.input_features, self.input_features)

     

    def forward(self, x, W_init = None):


        bsz = x.size(1)

        x = x.permute(1,0,2)

        x = x.contiguous().view(-1, bsz * self.nhead, self.head_dim).permute(1,0,2)

        #W_init = Variable(torch.eye(x.size(1)).unsqueeze(0).repeat(x.size(0), 1, 1).unsqueeze(3)).cuda()
        
        #W_init = W_init.cuda()

        W_new = self.module_w(x, W_init.unsqueeze(3))
        #W_new[W_new<0.1] = 0 

        Wi = torch.cat([W_init.unsqueeze(3), W_new], 3)

        x_new = F.leaky_relu(self.module_l([Wi, x])[1])
        
        x = self.fc(torch.cat([x, x_new], 2))

        return x_new, W_new.squeeze(-1), None



class EdgeConv(nn.Module):

    def __init__(self, d_model=512, nhead=8):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = self.d_model//self.nhead
        self.conv = nn.Sequential(nn.Conv2d(self.head_dim*2, self.head_dim, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(self.head_dim),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(self.head_dim, 1, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(1),
                                   nn.LeakyReLU(negative_slope=0.2))
    def get_graph_feature(self, x):

        batch_size = x.size(0)
        num_points = x.size(2)
        x = x.view(batch_size, -1, num_points)
        #if idx is None:
        #    idx = knn(x, k=num_points)   # (batch_size, num_points, k)

        idx = torch.arange((num_points)).repeat(batch_size,num_points,1).to('cuda')
        device = torch.device('cuda')

        idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

        idx = idx + idx_base

        idx = idx.view(-1)
    
        _, num_dims, _ = x.size()

        x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
        feature = x.view(batch_size*num_points, -1)[idx, :]
        feature = feature.view(batch_size, num_points, num_points, num_dims) 
        x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, num_points, 1)
        
        feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
      
        return feature 

    def forward(self, x):

        bsz = x.size(1)

        x = x.permute(1,0,2)

        x = x.contiguous().view(-1, bsz * self.nhead, self.head_dim).permute(1,2,0)

        eij = self.get_graph_feature(x)

        eij = self.conv(eij)

        x = eij.max(dim=-1, keepdim=False)[0].permute(0,2,1)

        #eij = eij.max(dim=1, keepdim=False)[0]
        eij = self.conv2(eij).squeeze(1)

        #eij = F.relu(eij)

        return x, eij

class EdgeConvKNN(nn.Module):

    def __init__(self, d_model=512, nhead=8):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = self.d_model//self.nhead
        self.conv = nn.Sequential(nn.Conv2d(self.head_dim*2, self.head_dim, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(self.head_dim),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(self.head_dim, 1, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(1),
                                   nn.LeakyReLU(negative_slope=0.2))
    def get_graph_feature(self, x, k=10):

        batch_size = x.size(0)
        num_points = x.size(2)
        x = x.view(batch_size, -1, num_points)
        
        if k==-1:
            knnidx = torch.arange((num_points)).repeat(batch_size,num_points,1).to('cuda')
            k = num_points
        else:
            knnidx = knn(x, k=k) # (batch_size, num_points, k)

        device = torch.device('cuda')

        idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

        idx = knnidx + idx_base

        idx = idx.view(-1)
    
        _, num_dims, _ = x.size()

        x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
        feature = x.view(batch_size*num_points, -1)[idx, :]
        feature = feature.view(batch_size, num_points, k, num_dims) 
        x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
        
        feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
      
        return feature, knnidx

    def get_masks(self, eij, knnidx):

        bsz, num_points, k = eij.shape

        if k == num_points:

            mask = torch.zeros((bsz, num_points, num_points)).to('cuda').bool()

            return eij, mask

        else:

            mask = torch.ones((bsz, num_points, num_points)).to('cuda')
            e_mask = torch.ones((bsz, num_points, num_points)).to('cuda')
            mask = mask.scatter_(2, knnidx, 0.).bool()
            eij = e_mask.scatter_(2, knnidx, eij)

            return eij, mask

    def forward(self, x, k = 10):

        bsz = x.size(1)

        x = x.permute(1,0,2)

        x = x.contiguous().view(-1, bsz * self.nhead, self.head_dim).permute(1,2,0)

        eij, knnidx = self.get_graph_feature(x, k = k)

        eij = self.conv(eij)

        x = eij.max(dim=-1, keepdim=False)[0].permute(0,2,1)

        #eij = eij.max(dim=1, keepdim=False)[0]
        eij = self.conv2(eij).squeeze(1)

        eij, mask = self.get_masks(eij, knnidx)

        eij = F.softmax(eij, dim=-1)

        return x, eij, mask
            

class dynamic_graph_module(nn.Module):

    def __init__(self, d_model=512, nhead=8):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = self.d_model//self.nhead
        self.fc1 = nn.Sequential(
            nn.Linear(self.head_dim, self.head_dim//4),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(self.head_dim, self.head_dim//4),
            nn.ReLU()
        )

        self.thr_fc = nn.Linear(self.head_dim, 1)
        self.bias_fc = nn.Linear(self.head_dim, 1)

        self.conv = nn.Sequential(nn.Conv2d(self.head_dim*2, self.head_dim, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(self.head_dim),
                                   nn.LeakyReLU(negative_slope=0.2))

    def forward(self, x, k = 10):

        bsz = x.size(1)

        x = x.permute(1,0,2)

        x = x.contiguous().view(-1, bsz * self.nhead, self.head_dim).permute(1,0,2)

        feat1 = self.fc1(x)
        feat2 = self.fc2(x)

        A = torch.bmm(feat1,feat2.transpose(2,1))

        A[A<0.2]=0

        #thr = self.thr_fc(x.mean(1))
        #bias = self.bias_fc(x.mean(1))

        #T = M.mean(dim=2,keepdim=True)*thr.unsqueeze(-1)+bias.unsqueeze(-1)

        #A = F.relu(M - T)

        mask_b = (A!=0.).float()

        alpha = F.softmax(A, dim=-1)*mask_b

        x = torch.bmm(alpha, x)

        #A = self.conv(A)torch.mm(yi, pi)

        #x = A.max(dim=-1, keepdim=False)[0].permute(0,2,1)

        return x, A, None
            



class DynamicGraphConvolution(nn.Module):
    def __init__(self, in_features, nhead):
        super(DynamicGraphConvolution, self).__init__()
        self.d_model = in_features
        self.nhead = nhead
        self.head_dim = self.d_model//self.nhead
        in_features = self.head_dim
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.conv_global = nn.Conv1d(in_features, in_features, 1)
        self.bn_global = nn.BatchNorm1d(in_features)
        self.relu = nn.LeakyReLU(0.2)

        self.conv_global_attn = nn.Conv1d(in_features*2, in_features, 1)
        self.dynamic_weight = nn.Conv1d(in_features, in_features, 1)

    def forward_static_gcn(self, x):
        x = self.static_adj(x.transpose(1, 2))
        x = self.static_weight(x.transpose(1, 2))
        return x

    def forward_construct_dynamic_graph(self, x):
        ### Model global representations ###
        x_glb = self.gap(x)
        x_glb = self.conv_global(x_glb)
        x_glb = self.bn_global(x_glb)
        x_glb = self.relu(x_glb)
        x_glb = x_glb.expand(x_glb.size(0), x_glb.size(1), x.size(2))
        
        ### Construct the dynamic correlation matrix ###
        x = torch.cat((x_glb, x), dim=1)
        x = self.conv_global_attn(x)
        dynamic_adj = torch.bmm(x.transpose(2,1),x)
        #dynamic_adj = torch.sigmoid(dynamic_adj)
        dynamic_adj = F.softmax(dynamic_adj, dim=-1)

        return dynamic_adj

    def forward_dynamic_gcn(self, x, dynamic_adj):
        x = torch.matmul(x, dynamic_adj)
        x = self.relu(x)
        x = self.dynamic_weight(x)
        x = self.relu(x)
        return x

    def forward(self, x):
        """ D-GCN module
        Shape: 
        - Input: (B, C_in, N) # C_in: 1024, N: num_classes
        - Output: (B, C_out, N) # C_out: 1024, N: num_classes
        """
        #out_static = self.forward_static_gcn(x)
        #x = x + out_static # residual
        x = x.contiguous().view(-1, x.size(1) * self.nhead, self.head_dim)
        x = x.permute(1,2,0)
        dynamic_adj = self.forward_construct_dynamic_graph(x)
        x = self.forward_dynamic_gcn(x, dynamic_adj)
        
        return x.permute(0,2,1), dynamic_adj, None


class Graph_conv_block(nn.Module):
    def __init__(self, input_dim, output_dim, use_bn=True):
        super(Graph_conv_block, self).__init__()

        self.weight = nn.Linear(input_dim, output_dim)
        if use_bn:
            self.bn = nn.BatchNorm1d(output_dim)
        else:
            self.bn = None

    def forward(self, x, A):
        x_next = torch.matmul(A, x) # (b, N, input_dim)
        x_next = self.weight(x_next) # (b, N, output_dim)

        if self.bn is not None:
            x_next = torch.transpose(x_next, 1, 2) # (b, output_dim, N)
            x_next = x_next.contiguous()
            x_next = self.bn(x_next)
            x_next = torch.transpose(x_next, 1, 2) # (b, N, output)

        return x_next

class Adjacency_layer(nn.Module):
    def __init__(self, input_dim, hidden_dim, ratio=[2,2,1,1]):

        super(Adjacency_layer, self).__init__()

        module_list = []

        for i in range(len(ratio)):
            if i == 0:
                module_list.append(nn.Conv2d(input_dim, hidden_dim*ratio[i], 1, 1))
            else:
                module_list.append(nn.Conv2d(hidden_dim*ratio[i-1], hidden_dim*ratio[i], 1, 1))

            module_list.append(nn.BatchNorm2d(hidden_dim*ratio[i]))
            module_list.append(nn.LeakyReLU())

        module_list.append(nn.Conv2d(hidden_dim*ratio[-1], 1, 1, 1))

        self.module_list = nn.ModuleList(module_list)

    def forward(self, x):

        X_i = x.unsqueeze(2) # (b, N , 1, input_dim)
        X_j = torch.transpose(X_i, 1, 2) # (b, 1, N, input_dim)

        phi = torch.abs(X_i - X_j) # (b, N, N, input_dim)

        phi = torch.transpose(phi, 1, 3) # (b, input_dim, N, N)

        A = phi

        for l in self.module_list:
            A = l(A)
        # (b, 1, N, N)

        A = torch.transpose(A, 1, 3) # (b, N, N, 1)

        A = F.softmax(A, 2) # normalize

        return A.squeeze(3) # (b, N, N)

class GNN_v2(nn.Module):
    def __init__(self, d_model, nhead):

        super(GNN_v2, self).__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = self.d_model//self.nhead   

        ratio = [2, 2, 1, 1]
        #ratio = [2, 1]

        last_adjacency = Adjacency_layer(
                    input_dim=self.head_dim, 
                    hidden_dim=self.head_dim, 
                    ratio=ratio)

        last_conv = Graph_conv_block(
                input_dim=self.head_dim, 
                output_dim=self.head_dim,
                use_bn=False)


        self.last_adjacency = last_adjacency
        self.last_conv = last_conv


    def forward(self, x):

        bsz = x.size(1)
      
        x = x.permute(1,0,2)

        x = x.contiguous().view(-1, bsz * self.nhead, self.head_dim).permute(1,0,2)

        A = self.last_adjacency(x)
        out = self.last_conv(x, A)   

        return out, A, None


class Transformer(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, pos_embed):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        mask = mask.flatten(1)

        tgt = torch.zeros_like(query_embed)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        hs = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                          pos=pos_embed, query_pos=query_embed)
        return hs.transpose(1, 2), memory.permute(1, 2, 0).view(bs, c, h, w)


class RETransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                rel_pos: Optional[Tensor] = None):
        output = src
        edge = src
        
        for layer in self.layers:
            output, edge = layer(output, edge,  src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos, rel_pos=rel_pos)

        if self.norm is not None:
            output = self.norm(output)

        return output



class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src
        edge = src
        
        for layer in self.layers:
            output, edge = layer(output, edge,  src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt

        intermediate = []

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output.unsqueeze(0)


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class RETransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()


        self.d_model = d_model
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.gnn = GNN(d_model, nhead)#DynamicGraphConvolution(d_model, nhead)#EdgeConvKNN(d_model, nhead)#GNN(d_model, nhead)#
        self.WGs = clones(nn.Linear(64, 1, bias=True), nhead)
        # Implementation of Feedforward model
        self.linear = nn.Linear(d_model, d_model)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.lineare1 = nn.Linear(d_model, dim_feedforward)
        self.dropoute = nn.Dropout(dropout)
        self.lineare2 = nn.Linear(dim_feedforward, d_model)

        self.norme1 = nn.LayerNorm(d_model)
        self.norme2 = nn.LayerNorm(d_model)
        self.dropoute1 = nn.Dropout(dropout)
        self.dropoute2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None, 
                     ):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src, edge,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    rel_pos: Optional[Tensor] = None):

        bsz = src.shape[1]

        src2 = self.norm1(src)
        edge2 = edge
        #edge2 = self.norme1(edge)

        q = k = self.with_pos_embed(src2, pos)

        value, dot_attn_weights = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)

        #edge2 = src2 + edge2 
        #edge2 = self.with_pos_embed(edge2, pos)
        #relative_geometry_weights_per_head = F.relu(torch.cat([l(rel_pos) for l in self.WGs],0).squeeze(-1))

        #edge2, eij, mask = self.gnn(edge2, relative_geometry_weights_per_head)#self.gnn(edge2, k = 15)

        #dot_attn_weights = dot_attn_weights.masked_fill(mask, float('-inf'))

        #

        combined_attn_weights = dot_attn_weights# + torch.log(torch.clamp(eij, min=1e-6))#
        combined_attn_weights = torch.nn.Softmax(dim=-1)(combined_attn_weights) 

        attn_output = torch.bmm(combined_attn_weights, value) #+ edge2
        src2 = self.linear(attn_output.transpose(0, 1).contiguous().view(-1, bsz, self.d_model))
        edge2 = edge2.transpose(0, 1).contiguous().view(-1, bsz, self.d_model)


        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)


        edge = edge + self.dropoute1(edge2)
        #edge2 = self.norme2(edge)
        #edge2 = self.lineare2(self.dropoute(self.activation(self.lineare1(edge2))))
        #edge = edge + self.dropout2(edge2)


        return src, edge

    def forward(self, src, edge,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                rel_pos: Optional[Tensor] = None,):
        if self.normalize_before:
            return self.forward_pre(src, edge, src_mask, src_key_padding_mask, pos, rel_pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)



class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()


        self.d_model = d_model
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.gnn = GNN(d_model, nhead)#DynamicGraphConvolution(d_model, nhead)#EdgeConvKNN(d_model, nhead)#GNN(d_model, nhead)#
        # Implementation of Feedforward model
        self.linear = nn.Linear(d_model, d_model)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.lineare1 = nn.Linear(d_model, dim_feedforward)
        self.dropoute = nn.Dropout(dropout)
        self.lineare2 = nn.Linear(dim_feedforward, d_model)

        self.norme1 = nn.LayerNorm(d_model)
        self.norme2 = nn.LayerNorm(d_model)
        self.dropoute1 = nn.Dropout(dropout)
        self.dropoute2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src, edge,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):

        bsz = src.shape[1]

        src2 = self.norm1(src)
        edge2 = edge
        #edge2 = self.norme1(edge)

        q = k = self.with_pos_embed(src2, pos)

        value, dot_attn_weights = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)

        #edge2 = src2 + edge2 
        #edge2 = self.with_pos_embed(edge2, pos)

        #edge2, eij, mask = self.gnn(edge2)#self.gnn(edge2, k = 15)

        #dot_attn_weights = dot_attn_weights.masked_fill(mask, float('-inf'))

        combined_attn_weights =  dot_attn_weights# + torch.log(torch.clamp(eij, min=1e-6))#
        combined_attn_weights = torch.nn.Softmax(dim=-1)(combined_attn_weights) 

        attn_output = torch.bmm(combined_attn_weights, value) #+ edge2
        src2 = self.linear(attn_output.transpose(0, 1).contiguous().view(-1, bsz, self.d_model))
        edge2 = edge2.transpose(0, 1).contiguous().view(-1, bsz, self.d_model)


        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)


        edge = edge + self.dropoute1(edge2)
        #edge2 = self.norme2(edge)
        #edge2 = self.lineare2(self.dropoute(self.activation(self.lineare1(edge2))))
        #edge = edge + self.dropout2(edge2)


        return src, edge

    def forward(self, src, edge,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, edge, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()

        self.d_model = d_model
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear = nn.Linear(d_model, d_model)
        self.linearx = nn.Linear(d_model, d_model)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):

        bsz = tgt.shape[1]
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)

        value, dot_attn_weights = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)
        attn_weights = torch.nn.Softmax(dim=-1)(dot_attn_weights)
        attn_output = torch.bmm(attn_weights, value)
        tgt2 = self.linear(attn_output.transpose(0, 1).contiguous().view(-1, bsz, self.d_model))

        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)

        value, dot_attn_weights = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)
        attn_weights = torch.nn.Softmax(dim=-1)(dot_attn_weights)
        attn_output = torch.bmm(attn_weights, value)
        tgt2 = self.linearx(attn_output.transpose(0, 1).contiguous().view(-1, bsz, self.d_model))


        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_transformer(args):
    return Transformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
    )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")



# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Various positional encodings for the transformer.
"""


#from util.misc import NestedTensor
import math

class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len = 160):
        super().__init__()
        self.d_model = d_model
        
        # create constant 'pe' matrix with values dependant on 
        # pos and i
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = \
                math.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, i + 1] = \
                math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))
                
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
 
    
    def forward(self, x):
        # make embeddings relatively larger
        x = x * math.sqrt(self.d_model)
        #add constant to embedding
        seq_len = x.size(1)
        y = Variable(self.pe[:,:seq_len], \
        requires_grad=False)
        return y