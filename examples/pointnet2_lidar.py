import os.path as osp
import os, sys
import numpy as np
import argparse
import torch
import torch.nn.functional as F
from torch_geometric.datasets import ShapeNet
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
from torch_geometric.nn import knn_interpolate
from torch_geometric.utils import intersection_and_union as i_and_u
from dataset import Dataset
from pointnet2_classification import SAModule, GlobalSAModule, MLP
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
# train_loader = DataLoader(train_dataset, batch_size=12, shuffle=True,
#                           num_workers=6)
# test_loader = DataLoader(test_dataset, batch_size=12, shuffle=False,
#                          num_workers=6)


class FPModule(torch.nn.Module):
    def __init__(self, k, nn):
        super(FPModule, self).__init__()
        self.k = k
        self.nn = nn

    def forward(self, x, pos, batch, x_skip, pos_skip, batch_skip):
        x = knn_interpolate(x, pos, pos_skip, batch, batch_skip, k=self.k)
        if x_skip is not None:
            x = torch.cat([x, x_skip], dim=1)
        x = self.nn(x)
        return x, pos_skip, batch_skip


class Net(torch.nn.Module):
    def __init__(self, num_classes):
        super(Net, self).__init__()
        self.sa1_module = SAModule(0.04, 1, MLP([3 + 3, 64, 64, 128]))
        self.sa2_module = SAModule(0.1, 5, MLP([128 + 3, 128, 128, 256]))
        self.sa3_module = GlobalSAModule(MLP([256 + 3, 256, 512, 1024]))

        self.fp3_module = FPModule(1, MLP([1024 + 256, 256, 256]))
        self.fp2_module = FPModule(3, MLP([256 + 128, 256, 128]))
        self.fp1_module = FPModule(3, MLP([128 + 3, 128, 128, 128]))

        self.lin1 = torch.nn.Linear(128, 128)
        self.lin2 = torch.nn.Linear(128, 128)
        self.lin3 = torch.nn.Linear(128, num_classes)

    def forward(self, data):
        sa0_out = (data.x, data.pos, data.batch)
        sa1_out = self.sa1_module(*sa0_out)
        # print(sa1_out[0].size())
        sa2_out = self.sa2_module(*sa1_out)
        # print(sa2_out[0].size())
        sa3_out = self.sa3_module(*sa2_out)

        fp3_out = self.fp3_module(*sa3_out, *sa2_out)
        fp2_out = self.fp2_module(*fp3_out, *sa1_out)
        x, _, _ = self.fp1_module(*fp2_out, *sa0_out)

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin3(x)
        return F.log_softmax(x, dim=-1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net(3).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


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
        points = torch.from_numpy(point)#.cuda()
        labels = labels.long()
        points = points.float()
        datalist = []
        datalist.append(Data(pos=points[:, :3], x=points[:, 3:], y=labels).to(device))

        data_batch = Batch.from_data_list(datalist)
        data_batch.to(device)
        optimizer.zero_grad()
        out = model(data_batch)
        # print(out.size())
        labels = labels.view(-1, 1)[:, 0]
        loss = F.nll_loss(out, labels)

        loss.backward()
        optimizer.step()
        # total_loss += loss.item()
        # correct_nodes += out.max(dim=1)[1].eq(data.y).sum().item()
        # total_nodes += data.num_nodes

        if (i + 1) % 10 == 0:
            print(precision_score(batch_label, pred_choice, average=None))
            print(recall_score(batch_label, pred_choice, average=None))
            print(f1_score(batch_label, pred_choice, average=None))
            print('[{}/{}] Loss: {:.4f}'.format(
                i + 1, len(datasets_th), loss))
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