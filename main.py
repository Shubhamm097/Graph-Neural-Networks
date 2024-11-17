import matplotlib.pyplot as plt
import networkx as nx
from GCN import GCN
from GAT import GAT
import torch

from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.datasets import Planetoid

import argparse

# Usage: python main.py --network_type 'GCN'

# Observations for GCN
# Experiment Setting: 

#   Number of Epochs: 100
#   Hidden Channels: 16
# Dataset Name: CORA, Test Accuracy Observed: 0.79
# Dataset Name: CITESEER, Test Accuracy Observed: 0.71
# Dataset Name: PUBMED, Test Accuracy Observed: 0.78

# Observations for GAT
#  Experiment Setting:
#   Number of Epochs: 100
#   Hidden Channels: 16
#   Number of Heads: 8
# Dataset Name: CORA, Test Accuracy Observed: 0.80
# Dataset Name: CITESEER, Test Accuracy Observed: 0.68
# Dataset Name: PUBMED, Test Accuracy Observed:

def train(data, model, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

    return loss

def test(model, data):
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    test_correct = pred[data.test_mask] == data.y[data.test_mask]
    test_accuracy = int(test_correct.sum()) / int(data.test_mask.sum())

    return test_accuracy

def get_results(network):

    dataset_names = ['Cora', 'Citeseer', 'Pubmed']

    for i in range(len(dataset_names)):

        print(f'-------- {dataset_names[i]} --------')

        dataset = Planetoid(root='Citation_Datasets', name=dataset_names[i], transform=NormalizeFeatures())

        data = dataset[0]
        print(f'Dataset: {data}')

        hidden_channels = 16

        if network == 'GCN':
            model = GCN(hidden_channels, dataset)
            print(model)

        else:
            heads = 8
            model = GAT(dataset, hidden_channels, heads)
            print(model)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        criterion = torch.nn.CrossEntropyLoss()     

        for epoch in range(1, 101):
            loss = train(data, model, optimizer, criterion)
            print(f'Epoch: {epoch}, Loss: {loss:.4f}')


        test_accuracy = test(model, data)
        print(f'Test Accuracy: {test_accuracy:.4f}')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Specify the network type (GCN or GAT)')
    parser.add_argument('--network_type', type=str, required=True)

    args = parser.parse_args()

    network_type = args.network_type

    get_results(network_type)
