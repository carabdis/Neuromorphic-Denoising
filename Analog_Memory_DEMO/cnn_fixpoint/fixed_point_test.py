import os
import glob
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from model_fixed import Net
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))


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


def quantize_aware_training(model, device, train_loader, optimizer, epoch, save_file, min_loss):
    # lossLayer = torch.nn.BCEWithLogitsLoss()
    lossLayer = torch.nn.BCELoss()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        output = model.quantize_forward(data)
        loss = lossLayer(output, target)
        if loss.item() < min_loss:
            min_loss = loss.item()
            print('loss: %.3f' % loss.item())
            torch.save(model.state_dict(), save_file)
        print('loss: %.3f' % loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # if batch_idx % 15 == 0 and batch_idx > 0:
        #     print('Train Epoch: {} [{}/{}]\tLoss: {:.6f}'.format(
        #         epoch, batch_idx * len(data), len(train_loader.dataset), loss.item()
        #     ))

    return min_loss


def full_inference(model, test_loader):
    correct = 0
    for i, (data, target) in enumerate(test_loader, 1):
        data, target = data.to(device), target.to(device)
        output = model(data)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
    print('\nTest set: Full Model Accuracy: {:.0f}%\n'.format(100. * correct / len(test_loader.dataset)))


def quantize_inference(model, test_loader):
    correct = 0
    for i, (data, target) in enumerate(test_loader, 1):
        data, target = data.to(device), target.to(device)
        output = model.quantize_inference(data)
        # print('out', output[output>0.5])
        output = output.cpu().detach().numpy()
        target = target.cpu().detach().numpy()

        print(output[0])
        print(output[2411])
        print(output[2408])
        '''
        temp = output.reshape(-1)
        x_axis = np.arange(0, temp.shape[0])
        plt.scatter(x_axis, output)
        plt.title('output')
        plt.show()
        '''
        post_process(output, target)


def createFolderIfNotExists(folderOut):
    ''' creates folder if doesnt already exist
    warns if creation failed '''
    if not os.path.exists(folderOut):
        try:
            os.mkdir(folderOut)
        except OSError:
            print("Creation of the directory %s failed" % folderOut)


def post_process(y_pred, labels):
    # post_process
    window_length = 5
    ma_preds = np.zeros(y_pred.shape)
    nums = len(y_pred) - window_length + 1
    for j in range(nums):  # moving average
        avg = np.average(y_pred[j:j + window_length])
        ma_preds[window_length - 1 + j] = avg
    ma_preds = np.reshape(np.asarray(ma_preds > 0.75, int), (-1,))

    y_pred = ma_preds

    # evaluation
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

    print("TP, FP, TN, FN", TP, FP, TN, FN)
    print("accuracy", (TP + TN) / (TP + FP + TN + FN))


if __name__ == "__main__":
    batch_size = 64
    test_batch_size = 64
    seed = 7
    epochs = 200
    lr = 0.001
    momentum = 0.5
    save_model = True
    # torch.manual_seed(seed)
    # load_quant_model_file = None
    # load_quant_model_file = "ckpt/1d_cnn_qat.pt"

    folderOut = 'D:/PycharmProjects/1d_cnn_fixpoint/1D_CNN_dataset'
    pat_path = folderOut + '/chb' + '01'
    mean_se = np.load(pat_path + "/mean_stat.npy")
    std_se = np.load(pat_path + "/std_stat.npy")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    createFolderIfNotExists(folderOut + '/chb' + '01' + '/results' + str(0))
    folderIn = folderOut + '/chb' + '01' + '/train' + str(0)
    train_dataset = EEGDataset(folderIn, train=True, mean=mean_se, std=std_se)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)
    folderIn = folderOut + '/chb' + '01' + '/test' + str(0)
    test_dataset = EEGDataset(folderIn, train=False, mean=mean_se, std=std_se)
    test_loader = DataLoader(dataset=test_dataset, batch_size=len(test_dataset), shuffle=False)

    model = Net().to(device)
    #model.load_state_dict(torch.load('ckpt/1d_cnn.pt', map_location='cpu'))

    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.eval()

    #full_inference(model, test_loader)  # 全精度推理

    '''
    量化权重
    '''
    Qn = 2
    num_bits = 4
    '''
    量化卷积输入
    '''
    Qn_io = 2
    num_bits_io = 4

    model.quantize(Qn, num_bits, Qn_io, num_bits_io)
    print('Quantization bit: %d Fix point: %d' % (num_bits, Qn))
    #save_file = 'ckpt/1d_cnn_fix_' + str(num_bits) + 'bit.pt'

    # if load_quant_model_file is not None:
    #     model.load_state_dict(torch.load(load_quant_model_file))
    #     print("Successfully load quantized model %s" % load_quant_model_file)



    #model.freeze()

    load_file = 'ckpt/1d_cnn_fix_' + str(num_bits) + 'bit.pt'
    # torch.save(model.state_dict(), save_file)

    model.load_state_dict(torch.load(load_file))
    quantize_inference(model, test_loader)
