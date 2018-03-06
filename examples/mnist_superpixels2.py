import os
import sys

import torch
from torch import nn
import torch.nn.functional as F

sys.path.insert(0, '.')
sys.path.insert(0, '..')

from torch_geometric.datasets import MNISTSuperpixels2  # noqa
from torch_geometric.utils import DataLoader2  # noqa
from torch_geometric.transform import CartesianAdj  # noqa
from torch_geometric.nn.modules import SplineConv  # noqa
from torch_geometric.nn.functional import (sparse_voxel_max_pool,
                                           dense_voxel_max_pool)  # noqa

path = os.path.dirname(os.path.realpath(__file__))
path = os.path.join(path, '..', 'data', 'MNISTSuperpixels2')

transform = CartesianAdj()
train_dataset = MNISTSuperpixels2(path, True, transform=transform)
test_dataset = MNISTSuperpixels2(path, False, transform=transform)
train_loader = DataLoader2(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader2(test_dataset, batch_size=64)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = SplineConv(1, 32, dim=2, kernel_size=5)
        self.conv2 = SplineConv(32, 64, dim=2, kernel_size=5)
        self.conv3 = SplineConv(64, 128, dim=2, kernel_size=5)
        self.fc1 = nn.Linear(4 * 128, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, data):
        data.input = F.elu(self.conv1(data.adj, data.input))
        data, _ = sparse_voxel_max_pool(data, 5, 0, transform)
        data.input = F.elu(self.conv2(data.adj, data.input))
        data, _ = sparse_voxel_max_pool(data, 7, 0, transform)
        data.input = F.elu(self.conv3(data.adj, data.input))

        data, _ = dense_voxel_max_pool(data, 14, 0, 28, transform)

        x = data.input.view(-1, self.fc1.weight.size(1))
        x = F.elu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


model = Net()
if torch.cuda.is_available():
    model.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


def train(epoch):
    model.train()

    if epoch == 6:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.001

    if epoch == 16:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.0001

    for data in train_loader:
        data = data.cuda().to_variable()
        optimizer.zero_grad()
        F.nll_loss(model(data), data.target).backward()
        optimizer.step()


def test(epoch):
    model.eval()
    correct = 0

    for data in test_loader:
        data = data.cuda().to_variable('input')
        pred = model(data).data.max(1)[1]
        correct += pred.eq(data.target).sum()

    print('Epoch:', epoch, 'Test Accuracy:', correct / len(test_dataset))


for epoch in range(1, 21):
    train(epoch)
    test(epoch)