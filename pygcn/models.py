#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   models.py
@Author  :   yyhaker 
@Contact :   572176750@qq.com
@Time    :   2020/03/04 15:06:06
'''

# here put the import lib
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution

from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv

class GCN(nn.Module):
    '''pure GCN just use PyTorch'''
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)


class KipfGCN(nn.Module):
    '''GCN use torch_geometric framework'''
    def __init__(self, data, num_class, args):
        super(KipfGCN, self).__init__()
        self.args = args
        self.data = data
        self.conv1 = GCNConv(self.data.num_features, self.args.gcn_dim, cached=True)
        self.conv2 = GCNConv(self.args.gcn_dim, num_class, cached=True)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.args.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)