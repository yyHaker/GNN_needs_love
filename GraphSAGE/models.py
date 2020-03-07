#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   models.py
@Author  :   yyhaker 
@Contact :   572176750@qq.com
@Time    :   2020/03/07 11:22:37
'''

# here put the import lib
import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class SupervisedGraphSage(nn.Module):
    '''Simple supervised GraphSAGE model.Here is implemented by pure PyTorch.'''
    def __init__(self, num_classes, encoder):
        super(SupervisedGraphSage, self).__init__()
        self.encoder = encoder
        self.xent = nn.CrossEntropyLoss()

        self.weight = nn.Parameter(torch.FloatTensor(num_classes, encoder.embed_dim))
        nn.init.xavier_uniform(self.weight)

    def forward(self, nodes):
        embeds = self.encoder(nodes)
        scores = self.weight.mm(embeds)
        return scores.t()

    def loss(self, nodes, labels):
        scores = self.forward(nodes)
        return self.xent(scores, labels.squeeze())


class Encoder(nn.Module):
    """
    Encodes a node's using 'convolutional' GraphSage approach.
    """
    def __init__(self, features, feature_dim, 
            embed_dim, adj_lists, aggregator,
            num_sample=10, base_model=None,
            gcn=False, cuda=False, 
            feature_transform=False): 
        super(Encoder, self).__init__()

        self.features = features
        self.feat_dim = feature_dim
        self.adj_lists = adj_lists

        self.embed_dim = embed_dim

        self.aggregator = aggregator
        self.num_sample = num_sample
        if base_model != None:
            self.base_model = base_model

        self.gcn = gcn
        self.cuda = cuda
        self.aggregator.cuda = cuda

        self.weight = nn.Parameter(
                torch.FloatTensor(embed_dim, self.feat_dim if self.gcn else 2 * self.feat_dim))
        nn.init.xavier_uniform(self.weight)

    def forward(self, nodes):
        """
        Generates embeddings for a batch of nodes.
        nodes: list of nodes
        """
        neigh_feats = self.aggregator.forward(nodes, [self.adj_lists[int(node)] for node in nodes], 
                self.num_sample)
        # if use gcn
        if self.gcn:
            combined = neigh_feats
        else:
            if self.cuda:
                self_feats = self.features(torch.LongTensor(nodes).cuda())
            else:
                self_feats = self.features(torch.LongTensor(nodes))
            combined = torch.cat([self_feats, neigh_feats], dim=1)
            
        combined = F.relu(self.weight.mm(combined.t()))
        return combined


class GraphSAGE(nn.Module):
    '''GraphSAGE model use torch_geometric'''
    def __init__(self, data, num_classes, args):
        super(GraphSAGE, self).__init__()
        self.args = args
        self.data = data
        self.conv1 = SAGEConv(self.data.num_features, self.args.sage_hidden, normalize=False)
        self.conv2 = SAGEConv(self.args.sage_hidden, num_classes, normalize=False)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.args.dropout, training=self.training)
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.args.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)
