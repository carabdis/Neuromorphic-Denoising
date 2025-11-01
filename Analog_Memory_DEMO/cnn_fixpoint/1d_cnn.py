'''
Author: CSuperlei
Date: 2022-10-02 10:47:48
LastEditTime: 2022-10-10 22:26:20
Description: 
'''
import os
import glob
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from model_fixed import Net


class EEGDataset(Dataset):
    def __init__(self, root, mean, std, train=True):
        super(EEGDataset, self).__init__()
        self.train = train
        file_path = root

        self.x = []
        self.y = []
        x_path = sorted(glob.glob(f'{file_path}/*s_norm.npy'))
        # print(x_path)
        y_path = sorted(glob.glob(f'{file_path}/*label.npy'))
        # print(y_path)
        for i, path in enumerate(x_path):
            if i == 0:
                self.x = np.load(path)
                self.y = np.load(y_path[i])
            else:
                self.x = np.concatenate((self.x, np.load(path)))
                self.y = np.concatenate((self.y, np.load(y_path[i])))

        # self.x = (self.x - mean) / std

        self.x = torch.from_numpy(self.x).to(torch.float32)
        self.x = self.x.permute(0, 2, 1)
        self.y = torch.from_numpy(self.y).to(torch.float32)

    def __getitem__(self, index):
        x_show = self.x[index]
        y_show = self.y[index]

        return x_show, y_show

    def __len__(self):
        return len(self.x)


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    lossLayer = torch.nn.BCEWithLogitsLoss()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = lossLayer(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % 15 == 0 and batch_idx > 0:
            print('Loss:', loss.item())


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    lossLayer = torch.nn.BCEWithLogitsLoss()
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        target = target * 1e2
        test_loss += lossLayer(output, target).item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {:.0f}%\n'.format(
        test_loss, 100. * correct / len(test_loader.dataset)
    ))


def createFolderIfNotExists(folderOut):
    ''' creates folder if doesnt already exist
    warns if creation failed '''
    if not os.path.exists(folderOut):
        try:
            os.mkdir(folderOut)
        except OSError:
            print("Creation of the directory %s failed" % folderOut)


if __name__ == "__main__":
    batch_size = 64
    test_batch_size = 64
    seed = 7
    epochs = 300
    lr = 0.01
    momentum = 0.5
    save_model = True
    torch.manual_seed(seed)

    folderOut = 'C:\\Users\\16432\\Desktop\\Workplace\\Python\\cnn_fixpoint\\1D_CNN_dataset'
    pat_path = folderOut + '/chb' + '02'
    mean_se = np.load(pat_path + "\\mean_stat.npy")
    std_se = np.load(pat_path + "\\std_stat.npy")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    createFolderIfNotExists(folderOut + '\\chb' + '02' + '\\results' + str(0))
    folderIn = folderOut + '\\chb' + '02' + '\\train' + str(0)
    train_dataset = EEGDataset(folderIn, train=True, mean=mean_se, std=std_se)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)

    folderIn = folderOut + '\\chb' + '02' + '\\test' + str(0)
    test_dataset = EEGDataset(folderIn, train=False, mean=mean_se, std=std_se)
    test_loader = DataLoader(dataset=test_dataset, batch_size=len(test_dataset), shuffle=False)

    model = Net().to(device)
    # model.load_state_dict(torch.load(('C:\\Users\\16432\\Desktop\\Workplace\\Python\\cnn_fixpoint\\ckpt\\1d_cnn.pt'), map_location='cpu'))
    optimizer = optim.Adam(model.parameters(), lr=lr)  # default lr=0.001
    model.quantize(0, 2, 0, 2)
    model.load_state_dict(torch.load(('C:\\Users\\16432\\Desktop\\Workplace\\Python\\cnn_fixpoint\\ckpt\\1d_cnn_fix_2bit_sdj.pt'), map_location='cpu'))
    test(model, device, test_loader)
    # for epoch in range(1, epochs + 1):
    #     train(model, device, train_loader, optimizer, epoch)
    #     if epoch % 20 == 0:
    #         print("Epoch", epoch)
    #         test(model, device, test_loader)
        

    # if save_model:
    #     if not os.path.exists('C:\\Users\\16432\\Desktop\\Workplace\\Python\\cnn_fixpoint\\ckpt'):
    #         os.makedirs('C:\\Users\\16432\\Desktop\\Workplace\\Python\\cnn_fixpoint\\ckpt')
    #     else:
    #         torch.save(model.state_dict(), 'C:\\Users\\16432\\Desktop\\Workplace\\Python\\cnn_fixpoint\\ckpt\\1d_cnn_sdj_2bit_new.pt')
