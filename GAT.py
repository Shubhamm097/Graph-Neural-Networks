import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
from torch_geometric.nn import GATConv

class GAT(nn.Module):

    def __init__(self, dataset, hidden_channels, heads):
        super().__init__()
        torch.manual_seed(12)
        self.conv1 = GATConv(in_channels=dataset.num_features, out_channels=hidden_channels, heads=heads)
        self.conv2 = GATConv(in_channels=hidden_channels * heads, out_channels=dataset.num_classes, heads=1)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)

        return x
