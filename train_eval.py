import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

# My model from model.py
from model import GCN

dataset = Planetoid(root = 'data/Planetoid', name='Cora', transform=NormalizeFeatures())
data = dataset[0]

# training definitions
model = GCN(hidden=16)
optimizer = torch.optim.Adam(model.parameters(),lr =0.01, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()


# train function
def train():
    model.train()
    optimizer.zero_grad() # 1. Clear Gradient
    out = model(data.x, data.edge_index) # perform a single forward pass 
    loss = criterion(out[data.train_mask], data.y[data.train_mask]) # Compute loss
    loss.backward() # backpropagate 
    optimizer.step()
    return loss

# test function
def test():
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)

    test_correct = pred[data.test_mask] == data.y[data.test_mask]
    test_acc = int(test_correct.sum()) / int(data.test_mask.sum())

    return test_acc

# training loop
for i in range(100):
    loss = train()
    print(f'Epoch: {i+1}, Loss: {loss:.4f}')

test_acc = test()
print(test_acc)