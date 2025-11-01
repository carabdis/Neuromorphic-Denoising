import numpy as np

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
from cnn_fixpoint.model_fixed import Net
# from model_fixed import Net
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1.0/(1+np.exp(-x))

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
            print('Train Epoch: {} [{}/{}]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset), loss.item()
            ))


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    lossLayer = torch.nn.BCEWithLogitsLoss()
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
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

def post_process(y_pred, labels):
    #post_process
    window_length = 5
    ma_preds = np.zeros(y_pred.shape)
    nums = len(y_pred) - window_length + 1
    for j in range(nums):  # moving average
        avg = np.average(y_pred[j:j + window_length])
        ma_preds[window_length - 1 + j] = avg
    ma_preds = np.reshape(np.asarray(ma_preds > 0.9, int), (-1,))

    y_pred = ma_preds

    #evaluation
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for m in range(len(y_pred)):
        if y_pred[m] == 1:
            if labels[m] == 1:
                TP = TP + 1
            else:
                FP = FP + 1
        else:
            if labels[m] == 1:
                FN = FN + 1
            else:
                TN = TN + 1


    # print("TP, FP, TN, FN", TP, FP, TN, FN)
    # print("accuracy", (TP+TN)/(TP+FP+TN+FN))
    # print("sensitivity", TP / (TP + FN))
    return TP, FP, TN, FN

if __name__ == "__main__":
    batch_size = 64
    test_batch_size = 64
    seed = 7
    epochs = 300
    lr = 0.05
    momentum = 0.5
    save_model = False
    #torch.manual_seed(seed)


    folderOut = 'C:\\Users\\16432\\Desktop\\Workplace\\Python\\cnn_fixpoint\\1D_CNN_dataset'
    pat_path = folderOut + '\\chb' + '01'
    mean_se = np.load(pat_path + "\\mean_stat.npy")
    std_se = np.load(pat_path + "\\std_stat.npy")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    folderIn = folderOut + '\\chb' + '01' + '\\test' + str(0)
    test_dataset = EEGDataset(folderIn, train=False, mean=mean_se, std=std_se)
    test_loader = DataLoader(dataset=test_dataset, batch_size=len(test_dataset), shuffle=False)

    model = Net().to(device)
    model.load_state_dict(torch.load(('C:\\Users\\16432\\Desktop\\Workplace\\Python\\cnn_fixpoint\\ckpt\\1d_cnn.pt'), map_location='cpu'))

    for i, (data, target) in enumerate(test_loader, 1):
        data, target = data.to(device), target.to(device)
        output = model(data)
        output = output.cpu().detach().numpy()
        output = sigmoid(output)
        post_process(output, target)
        """
        temp = output.reshape(-1)
        x_axis = np.arange(0, temp.shape[0])
        plt.scatter(x_axis, output)
        plt.title('test_data')
        plt.show()
        """

