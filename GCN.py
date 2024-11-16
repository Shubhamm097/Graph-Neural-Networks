import os
import math
import numpy as np
import time

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim

from torch_geometric.nn import GCNConv

import torchvision
from torchvision.datasets import CIFAR10
from torchvision import transforms

# import pytorch_lightning as pl
# from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

# class GCN(nn.Module):
    
#     def __init__(self, c_in, c_out):
#         super().__init__()
#         self.projection = nn.Linear(c_in, c_out)

#     def forward(self, node_feats, adj_matrix):

#         # node_feats = Tensor with node features of shape[batch_size, num_nodes, c_in]
#         # adj_matrix = shape: [batch_size, num_nodes, num_nodes]
#         # num_neighbors = Number of incoming edges

#         num_neighbors = adj_matrix.sum(dim=-1, keepdims=True)
#         node_feats = self.projection(node_feats)
#         node_feats = torch.bmm(adj_matrix, node_feats)
#         node_feats = node_feats / num_neighbors
#         return node_feats

class GCN(nn.Module):

    def __init__(self, hidden_channels, dataset):
        super().__init__()
        torch.manual_seed(12)
        self.conv1 = GCNConv(dataset.num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, dataset.num_classes)

    def forward(self, x, edge_index):

        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)

        return x