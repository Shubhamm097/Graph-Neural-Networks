import matplotlib.pyplot as plt
import networkx as nx
from GCN import GCN
import torch

from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.datasets import Planetoid

# Observations
# Dataset Name: CORA, Epochs: 100, Hidden Channels: 16, Test Accuracy Observed: 0.79
# Dataset Name: CITESEER, Epochs: 100, Hidden Channels: 16, Test Accuracy Observed: 0.71
# Dataset Name: PUBMED, Epochs: 100, Hidden Channels: 16, Test Accuracy Observed: 0.78

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

def get_results():

    dataset_names = ['Cora', 'Citeseer', 'Pubmed']

    for i in range(len(dataset_names)):

        print(f'-------- {dataset_names[i]} --------')

        dataset = Planetoid(root='PyTorch_Practice_Models', name=dataset_names[i], transform=NormalizeFeatures())

        data = dataset[0]
        print(f'Dataset: {data}')

        hidden_channels = 16

        model = GCN(hidden_channels, dataset)
        print(model)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        criterion = torch.nn.CrossEntropyLoss()     

        for epoch in range(1, 101):
            loss = train(data, model, optimizer, criterion)
            print(f'Epoch: {epoch}, Loss: {loss:.4f}')


        test_accuracy = test(model, data)
        print(f'Test Accuracy: {test_accuracy:.4f}')

get_results()
