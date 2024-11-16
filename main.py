import matplotlib.pyplot as plt
import networkx as nx
from GCN import GCN
import torch

from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.datasets import Planetoid

# nx_g = nx.from_edgelist([(2, 1), (2, 3), (4, 2), (3, 4)])

# node_feats = torch.arange(8, dtype=torch.float32).view(1, 4, 2)

# adj_matrix = torch.Tensor([
#     [[1, 1, 0, 0],
#      [1, 1, 1, 1],
#      [0, 1, 1, 1],
#      [0, 1, 1, 1]],
# ])

# print("Node features:\n", node_feats)
# print("\nAdjacency matrix:\n", adj_matrix)


dataset = Planetoid(root='PyTorch_Practice_Models', name='Cora', transform=NormalizeFeatures())

data = dataset[0]
print(f'Dataset: {data}')

hidden_channels = 16
lr = 0.01
weight_decay = 5e-4

model = GCN(hidden_channels, dataset)
print(model)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()

def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

    return loss

def test():
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    test_correct = pred[data.test_mask] == data.y[data.test_mask]
    test_accuracy = int(test_correct.sum()) / int(data.test_mask.sum())

    return test_accuracy

for epoch in range(1, 101):
    loss = train()
    print(f'Epoch: {epoch}, Loss: {loss:.4f}')


test_accuracy = test()
print(f'Test Accuracy: {test_accuracy:.4f}')