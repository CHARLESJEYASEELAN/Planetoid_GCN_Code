from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

dataset = Planetoid(root = 'data/Planetoid', name='Cora', transform=NormalizeFeatures())
data = dataset[0]

class GCN(torch.nn.Module):
    def __init__(self, hidden):
        super().__init__()
        torch.manual_seed(123)

        self.conv1 = GCNConv(dataset.num_features, hidden)
        self.conv2 = GCNConv(hidden, dataset.num_classes)

    def forward(self,x, edge_index):
        x = self.conv1(x,edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x,edge_index)
        return x 

model = GCN(hidden=16)
print(model)