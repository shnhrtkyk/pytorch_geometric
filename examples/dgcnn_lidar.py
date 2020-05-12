import os.path as osp
import os, sys
import numpy as np
import argparse
import torch
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Dropout, Linear as Lin
from torch_geometric.datasets import ShapeNet
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
from torch_geometric.nn import DynamicEdgeConv
from torch_geometric.utils import intersection_and_union as i_and_u
from dataset import Dataset
from pointnet2_classification import MLP
from torch_geometric.data import  Data, Batch

# transform = T.Compose([
#     T.RandomTranslate(0.01),
#     T.RandomRotate(15, axis=0),
#     T.RandomRotate(15, axis=1),
#     T.RandomRotate(15, axis=2)
# ])
# pre_transform = T.NormalizeScale()
# train_dataset = ShapeNet(path, category, split='trainval', transform=None,
#                          pre_transform=pre_transform)
# test_dataset = ShapeNet(path, category, split='test',
#                         pre_transform=pre_transform)
# train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True,
#                           num_workers=6)
# test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False,
#                          num_workers=6)


class Net(torch.nn.Module):
    def __init__(self, out_channels, k=3, aggr='max'):
        super(Net, self).__init__()

        self.conv1 = DynamicEdgeConv(MLP([2 * 6, 64, 64]), k, aggr)
        self.conv2 = DynamicEdgeConv(MLP([2 * 64, 64, 64]), k, aggr)
        self.conv3 = DynamicEdgeConv(MLP([2 * 64, 64, 64]), k, aggr)
        self.lin1 = MLP([3 * 64, 1024])

        self.mlp = Seq(MLP([1024, 256]), Dropout(0.5), MLP([256, 128]),
                       Dropout(0.5), Lin(128, out_channels))

    def forward(self, data):
        x, pos, batch = data.x, data.pos, data.batch
        x0 = torch.cat([x, pos], dim=-1)
        x1 = self.conv1(x0, batch)
        x2 = self.conv2(x1, batch)
        x3 = self.conv3(x2, batch)
        out = self.lin1(torch.cat([x1, x2, x3], dim=1))
        out = self.mlp(out)
        return F.log_softmax(out, dim=1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net(3, k=5).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.8)



def train(datasets_th):
    model.train()
    label = np.ones((200000, 1))
    point = np.ones((200000, 6))
    total_loss = correct_nodes = total_nodes = 0
    # for i, data in enumerate(train_loader):
    for i in range(len(datasets_th)):
        file_in = datasets_th[i]
        # print(file_in)
        if isinstance(file_in, Dataset):
            ds = file_in
        else:
            ds = Dataset(file_in)
        label[:, 0] = ds.labels
        point[:, :] = ds.points_and_features[:, :]
        labels = torch.from_numpy(label).cuda()
        points = torch.from_numpy(point).cuda()
        labels = labels.long()
        points = points.float()
        data = Data(pos=points[:, :3], x=points[:, 3:], y=labels)
        data.to(device)
        datalist = []
        datalist.append(data)

        data_batch = Batch.from_data_list(datalist)
        data_batch.to(device)
        optimizer.zero_grad()
        out = model(data_batch)
        labels = labels.view(-1, 1)[:, 0]
        loss = F.nll_loss(out, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct_nodes += out.max(dim=1)[1].eq(data.y).sum().item()
        total_nodes += data.num_nodes

        if (i + 1) % 10 == 0:
            print('[{}/{}] Loss: {:.4f}, Train Accuracy: {:.4f}'.format(
                i + 1, len(train_loader), total_loss / 10,
                correct_nodes / total_nodes))
            total_loss = correct_nodes = total_nodes = 0
            savepath = './model.pth'
            print('Saving at %f' % elapsed_time)
            state = {
                'epoch': j,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(state, savepath)


def test(loader):
    model.eval()

    correct_nodes = total_nodes = 0
    intersections, unions, categories = [], [], []
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            out = model(data)
        pred = out.max(dim=1)[1]
        correct_nodes += pred.eq(data.y).sum().item()
        total_nodes += data.num_nodes
        i, u = i_and_u(pred, data.y, test_dataset.num_classes, data.batch)
        intersections.append(i.to(torch.device('cpu')))
        unions.append(u.to(torch.device('cpu')))
        categories.append(data.category.to(torch.device('cpu')))

    category = torch.cat(categories, dim=0)
    intersection = torch.cat(intersections, dim=0)
    union = torch.cat(unions, dim=0)

    ious = [[] for _ in range(len(loader.dataset.categories))]
    for j in range(len(loader.dataset)):
        i = intersection[j, loader.dataset.y_mask[category[j]]]
        u = union[j, loader.dataset.y_mask[category[j]]]
        iou = i.to(torch.float) / u.to(torch.float)
        iou[torch.isnan(iou)] = 1
        ious[category[j]].append(iou.mean().item())

    for cat in range(len(loader.dataset.categories)):
        ious[cat] = torch.tensor(ious[cat]).mean().item()

    return correct_nodes / total_nodes, torch.tensor(ious).mean().item()


def main(args):
    inlist = args.inList
    threshold = args.threshold
    train_size = args.trainSize
    lr = args.learningRate
    normalize_vals = args.normalize == 1

    with open(inlist, "rb") as f:
        _ = f.readline()  # remove header
        rest = f.readlines()

    datasets = []
    all_ds = []
    for line in rest:
        line = line.decode('utf-8')
        linespl = line.split(",")
        dataset_path = os.path.join(os.path.dirname(inlist), linespl[0])
        print(linespl)
        datasets.append(dataset_path)
        all_ds.append(dataset_path)
    print(len(datasets))
    np.random.shuffle(datasets)
    datasets_th = []
    for idx, dataset in enumerate(datasets):
        print("Loading dataset %s of %s (%s)" % (idx + 1, len(datasets), os.path.basename(dataset)))
        # print(dataset)
        ds = Dataset(dataset, load=False, normalize=normalize_vals)
        datasets_th.append(ds)
    print("%s datasets loaded." % len(datasets_th))
    sys.stdout.flush()

    for epoch in range(1, 31):
        train(datasets_th)
    # acc, iou = test(test_loader)
    # print('Epoch: {:02d}, Acc: {:.4f}, IoU: {:.4f}'.format(epoch, acc, iou))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--inList', help='input text file, must be csv with filename;stddev;...')
    parser.add_argument('--threshold', default=20,  type=float, help='upper threshold for class stddev')
    parser.add_argument('--minBuild', default=0, type=float, help='lower threshold for buildings class [0-1]')
    parser.add_argument('--outDir', required=False, help='directory to write html log to')
    # parser.add_argument('--multiclass', default=True, type=bool, help='label into multiple classes ' +
    #                                                                  '(not only ground/nonground) [default: True]')
    parser.add_argument('--multiTrain', default=200, type=int,
                       help='how often to feed the whole training dataset [default: 1]')
    parser.add_argument('--trainSize', default=1, type=int,
                       help='how many plots to train at once [default: 1]')
    parser.add_argument('--learningRate', default=0.0005, type=float,
                       help='learning rate [default: 0.001]')
    parser.add_argument('--archFile', default="1", type=str,
                       help='architecture file to import [default: default architecture]')
    parser.add_argument('--continueModel', default=None, type=str,
                        help='continue training an existing model [default: start new model]')
    parser.add_argument('--lossFn', default='fp_high', type=str,
                        help='loss function to use [default: fp_high][simple/fp_high]')
    parser.add_argument('--normalize', default=1, type=int,
                        help='normalize fields and coordinates [default: 1][1/0]')
    parser.add_argument('--points', default=200000, type=int,
                        help='normalize fields and coordinates [default: 1][1/0]')
    parser.add_argument('--classes', default=6, type=int,
                        help='normalize fields and coordinates [default: 1][1/0]')
    # parser.add_argument('--testList', help='list with files to test on')
    parser.add_argument('--gpuID', default=0, help='which GPU to run on (default: CPU only)')
    parser.add_argument('--optimizer', default="Adam", help='which Optimizer (default: Adam)')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='weight decay [default: 1e-4]')
    args = parser.parse_args()
    main(args)