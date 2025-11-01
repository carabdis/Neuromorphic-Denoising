
from __future__ import division, with_statement, print_function
# from aardvark_api import *
from aardvark_api.python.aardvark_py import *
from itertools import filterfalse
from cnn_fixpoint.test import EEGDataset, sigmoid, post_process
from cnn_fixpoint.model_fixed_0429 import Net, FakeQuantize
import os
# import aardvark_api.python

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.autograd import Function
import numpy as np
import time

from tqdm import tqdm
from torchvision.utils import save_image, make_grid
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.nn.functional as F
import csv

# Model Hyperparameters

dataset_path = 'C:/Users/16432/Desktop/Workplace/Python/VAE_DEMO/datasets'

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
Prob = [[1.0, 0.0, 0.0, 0.0, ],
        [1.0, 0.0, 0.0, 0.0, ],
        [1.0, 0.0, 0.0, 0.0, ],
        [1.0, 0.0, 0.0, 0.0, ],
        [1.0, 0.0, 0.0, 0.0, ],
        [1.0, 0.0, 0.0, 0.0, ],
        [1.0, 0.0, 0.0, 0.0, ],
        [1.0, 0.0, 0.0, 0.0, ],
        [0.7777777777777778, 0.2222222222222222, 0.0, 0.0, ],
        [1.0, 0.0, 0.0, 0.0, ],
        [1.0, 0.0, 0.0, 0.0, ],
        [0.5, 0.5, 0.0, 0.0, ],
        [0.4, 0.6, 0.0, 0.0, ],
        [0.8, 0.2, 0.0, 0.0, ],
        [0.2, 0.8, 0.0, 0.0, ],
        [0.0, 1.0, 0.0, 0.0, ],
        [0.14285714285714285, 0.7142857142857143, 0.14285714285714285, 0.0, ],
        [0.2222222222222222, 0.7777777777777778, 0.0, 0.0, ],
        [0.2, 0.8, 0.0, 0.0, ],
        [0.0, 1.0, 0.0, 0.0, ],
        [0.0, 1.0, 0.0, 0.0, ],
        [0.0, 1.0, 0.0, 0.0, ],
        [0.0, 1.0, 0.0, 0.0, ],
        [0.0, 1.0, 0.0, 0.0, ],
        [0.0, 0.5, 0.5, 0.0, ],
        [0.0, 0.8, 0.2, 0.0, ],
        [0.0, 0.6153846153846154, 0.38461538461538464, 0.0, ],
        [0.0, 0.5, 0.5, 0.0, ],
        [0.0, 0.7142857142857143, 0.2857142857142857, 0.0, ],
        [0.0, 0.3333333333333333, 0.6666666666666666, 0.0, ],
        [0.0, 0.5, 0.5, 0.0, ],
        [0.0, 0.1111111111111111, 0.8888888888888888, 0.0, ],
        [0.0, 0.1111111111111111, 0.8888888888888888, 0.0, ],
        [0.0, 0.07692307692307693, 0.7692307692307693, 0.15384615384615385, ],
        [0.16666666666666666, 0.0, 0.8333333333333334, 0.0, ],
        [0.0, 0.0, 0.8571428571428571, 0.14285714285714285, ],
        [0.0, 0.0, 1.0, 0.0, ],
        [0.0, 0.0, 0.5, 0.5, ],
        [0.0, 0.0, 0.75, 0.25, ],
        [0.0, 0.16666666666666666, 0.8333333333333334, 0.0, ],
        [0.0, 0.0, 0.4, 0.6, ],
        [0.0, 0.0, 1.0, 0.0, ],
        [0.0, 0.0, 1.0, 0.0, ],
        [0.0, 0.0, 0.8, 0.2, ],
        [0.0, 0.0, 1.0, 0.0, ],
        [0.0, 0.0, 0.0, 1.0, ],
        [0.0, 0.0, 0.25, 0.75, ],
        [0.0, 0.0, 1.0, 0.0, ],
        [0.0, 0.0, 0.6666666666666666, 0.3333333333333333, ],
        [0.0, 0.0, 0.25, 0.75, ],
        [0.0, 0.0, 0.25, 0.75, ],
        [0.0, 0.0, 0.047619047619047616, 0.9523809523809523, ],
        [0.0, 0.0, 0.0, 1.0, ],
        [0.0, 0.0, 0.0, 1.0, ],
        [0.0, 0.0, 0.16666666666666666, 0.8333333333333334, ],
        [0.0, 0.0, 0.0, 1.0, ],
        [0.0, 0.0, 0.0, 1.0, ],
        [0.0, 0.0, 0.0, 1.0, ],
        [0.0, 0.0, 0.0, 1.0, ],
        [0.0, 0.0, 0.0, 1.0, ],
        [0.0, 0.0, 0.0, 1.0, ]]

ProbSample = torch.zeros((61, 1000), device=DEVICE)
for row in range(61):
    TempProb = torch.rand(1000)
    for col in range(1000):
        IntProb = 0.
        for key in range(4):
            if IntProb + Prob[row][key] > TempProb[col].item():
                ProbSample[row, col] = key
                break
            else:
                IntProb += Prob[row][key]

cuda = torch.cuda.is_available()
DEVICE = torch.device("cuda" if cuda else "cpu")
COMM_DEVICES = {
    'AT25080' : (1024, 32),
    'AT25256' : (32768, 64),
}
MAX_LIMIT = 200000
I2C_BITRATE =  100

batch_size = 1
# batch_size = 50

x_dim = 784
hidden_dim = 400
hidden_dim2 = 200
hidden_dim3 = 32
latent_dim = 16

lr = 1e-6

epochs = 200
fileA = "TestResult_CNN.txt"
fileA = open(fileA, "w")
fileB = "Deploy.txt"
fileB = open(fileB, "w")
# Load Dataset
mnist_transform = transforms.Compose([
    transforms.ToTensor(),
])

kwargs = {'num_workers': 8, 'pin_memory': True}

train_dataset = MNIST(dataset_path, transform=mnist_transform, train=True, download=True)
test_dataset = MNIST(dataset_path, transform=mnist_transform, train=True, download=True)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, **kwargs)

"""
    A simple implementation of Gaussian MLP Encoder and Decoder
"""

class Round(Function):
    @staticmethod
    def forward(self, input):
        output = torch.round(input)
        return output

    @staticmethod
    def backward(self, grad_output):
        grad_input = grad_output.clone()
        return grad_input

class Sign(Function):
    @staticmethod
    def forward(self, input):
        output = torch.sign(input)
        return output

    @staticmethod
    def backward(self, grad_output):
        grad_input = grad_output.clone()
        return grad_input


class QuantLinear(nn.Module):

    def __init__(self, Input_Dim, Output_Dim, Bias=False, InputQuant = False):
        super(QuantLinear, self).__init__()
        self.layer = nn.Linear(Input_Dim, Output_Dim, bias=Bias)
        self.scale = nn.Parameter(torch.tensor(15., device=DEVICE))
        self.InMax = nn.Parameter(torch.tensor(0., device=DEVICE))
        self.InMin = nn.Parameter(torch.tensor(10., device=DEVICE))
        self.m = torch.nn.ReLU()

    def forward(self, Input):
        w = Sign.apply(self.layer.weight)
        wPlus = self.m(w)
        wMinus = self.m(-w)
        # x = Input
        # print(self.max, self.min, "*****")
        # if self.training:
        #     self.InMax = nn.Parameter(torch.max(Input).detach())
        #     self.InMin = nn.Parameter(torch.min(Input).detach())
        Input = torch.clip(Round.apply((Input - self.InMin) / (self.InMax - self.InMin) * 15), 0, 15)
        # print(Input)
        # Input = (Input - self.InMin) / (self.InMax - self.InMin) * 15
        # Input = Input / 16 * (self.InMax - self.InMin)
        Min = torch.ones(Input.size(), device=DEVICE) * self.InMin
        bias = torch.nn.functional.linear(Min, w)
        # if self.training:
        #    self.max = torch.max(x).detach()
        #    self.min = torch.min(x).detach()
        OutList = []
        MaxList = []
        MinList = []
        # x = torch.nn.functional.linear(Input / 15 * (self.InMax - self.InMin), w) + bias
        x = torch.zeros((Input.size()[0], hidden_dim3), device=DEVICE)
        for i in range(4):
            # print(Input[:, (i * 4):(i + 1) * 4], "********************", file=fileB)
            # print(wPlus[:, (i * 4):(i + 1) * 4])
            # print(wMinus[:, (i * 4):(i + 1) * 4])
            y = torch.nn.functional.linear(Input[:, (i * 4):(i + 1) * 4], wPlus[:, (i * 4):(i + 1) * 4])
            OutList.append(y)
            # print(y, file=fileB)
            y = torch.nn.functional.linear(Input[:, (i * 4):(i + 1) * 4], wMinus[:, (i * 4):(i + 1) * 4])
            OutList.append(-y)
            # print(y, file=fileB)
            # y = torch.nn.functional.linear(Input[:, (i * 4):(i + 1) * 4], wPlus[:, (i * 4):(i + 1) * 4])
            # OutList.append(y)
            # y = torch.nn.functional.linear(Input[:, (i * 4):(i + 1) * 4], wMinus[:, (i * 4):(i + 1) * 4])
            # OutList.append(-y)
            # x = x + y
        # for i in range(16):
        #     for j in range(4):
        #         print(wPlus[i, (j * 4):(j + 1) * 4], file=fileB)
        #         print(wMinus[i, (j * 4):(j + 1) * 4], file=fileB)
        # x = x / 15 * (self.InMax - self.InMin) + bias
        # print(y.size())
        # if self.training:
        #     MaxList.append(torch.max(y).detach())
        #     MinList.append(torch.min(y).detach())
        # # if self.training:
        # #     self.max = nn.Parameter(max(MaxList))
        # #     self.min = nn.Parameter(min(MinList))
        # x = torch.zeros((Input.size()[0], hidden_dim3), device=DEVICE)
        # if self.training:
        #     tempA = torch.nn.functional.linear(Input, wPlus).detach()
        #     tempB = torch.nn.functional.linear(Input, wMinus).detach()
        #     self.scale = nn.Parameter(max(torch.max(tempA), torch.max(tempB)))
        for i in range(len(OutList)):
            # temp = tensor / 15
            # print(torch.max(temp), torch.min(temp), "********")
            tensor = OutList[i]
            # print(tensor, "*************")
            # print(torch.clip(Round.apply(tensor / self.scale), -4, 4))
            # if False:
            #     x = (x + torch.clip(tensor / self.scale, -4, 4) * self.scale / 15
            #          * (self.InMax - self.InMin))
            # else:
            # print(tensor, "###############")
            # print(torch.clip(Round.apply(tensor / self.scale), -4, 4))
            # x = (x + torch.clip(Round.apply(tensor / self.scale), -4, 4) * self.scale / 15
            #      * (self.InMax - self.InMin))
            x = (x + torch.clip(Round.apply(tensor / 15), -4, 4)
                 * (self.InMax - self.InMin))
        # print(x)
        x = x + bias
        return x

class Encoder(nn.Module):

    def __init__(self, Input_Dim, Hidden_Dim, Hidden_Dim2, Hidden_Dim3, Latent_Dim):
        super(Encoder, self).__init__()

        self.FC_input = nn.Linear(Input_Dim, Hidden_Dim)
        self.FC_input2 = nn.Linear(Hidden_Dim, Hidden_Dim2)
        self.FC_input3 = nn.Linear(Hidden_Dim2, Hidden_Dim3)
        self.FC_mean = nn.Linear(Hidden_Dim3, Latent_Dim)
        self.FC_var = nn.Linear(Hidden_Dim3, Latent_Dim)

        self.LeakyReLU = nn.LeakyReLU(0.2)

        self.training = True

    def forward(self, x):
        h_ = self.LeakyReLU(self.FC_input(x))
        h_ = self.LeakyReLU(self.FC_input2(h_))
        h_ = self.LeakyReLU(self.FC_input3(h_))
        Mean = self.FC_mean(h_)
        Log_Var = self.FC_var(h_)  # encoder produces mean and log of variance
        #             (i.e., parateters of simple tractable normal distribution "q"

        return Mean, Log_Var


class Decoder(nn.Module):
    def __init__(self, Latent_Dim, Hidden_Dim, Hidden_Dim2, Hidden_Dim3, Output_Dim):
        super(Decoder, self).__init__()
        self.FC_hidden = QuantLinear(Latent_Dim, Hidden_Dim3, Bias=False)
        # self.FC_hidden = nn.Linear(Latent_Dim, Hidden_Dim3, bias=False)
        self.FC_hidden2 = nn.Linear(Hidden_Dim3, Hidden_Dim2, bias=False)
        self.FC_hidden3 = nn.Linear(Hidden_Dim2, Hidden_Dim, bias=False)
        self.FC_output = nn.Linear(Hidden_Dim, Output_Dim, bias=False)

        # self.LeakyReLU = nn.LeakyReLU(0.2)
        self.LeakyReLU = nn.ReLU()

    def forward(self, x):
        h = self.LeakyReLU(self.FC_hidden(x))
        h = self.LeakyReLU(self.FC_hidden2(h))
        h = self.LeakyReLU(self.FC_hidden3(h))
        x_hat = torch.sigmoid(self.FC_output(h))
        # print("x_hat ", x_hat)
        # x_hat = Round.apply(x_hat)
        # print("x_hat after round ", x_hat)
        return x_hat


def reparameterization(Mean, Var):
    epsilon = torch.tensor(np.random.gamma(1, 1, Var.shape), dtype=torch.float).to(DEVICE)
    # epsilon = torch.randn_like(Var).to(DEVICE)  # sampling epsilon
    z = Mean + Var * epsilon  # reparameterization trick
    return z


class Model(nn.Module):
    def __init__(self, ENCODER, DECODER):
        super(Model, self).__init__()
        self.Encoder = ENCODER
        self.Decoder = DECODER

    def forward(self, x):
        Mean, Log_Var = self.Encoder(x)
        z = reparameterization(Mean, torch.exp(0.5 * Log_Var))  # takes exponential function (log var -> var)
        x_hat_active = self.Decoder(z)

        return x_hat_active, Mean, Log_Var


def loss_function(x_static, x_hat_static, mean_static, log_var_static):
    reproduction_loss = nn.functional.binary_cross_entropy(x_hat_static, x_static, reduction='sum')
    KLD = - 0.5 * torch.sum(1 + log_var_static - mean_static.pow(2) - log_var_static.exp())

    return reproduction_loss + KLD


def show_image(x_static, idx):
    x_static = x_static.view(28, 28 * 2)
    plt.imshow(x_static.cpu().numpy())
    plt.show()

def Communication_test():
    handle, bitrate = Communication_Init()
    print("Bitrate set to %d kHz" % bitrate)
    readin = array('B', [0x01])
    ENABLE_BYTE = 0x00
    # ENABLE_BYTE = 0x01
    ADDR_BYTE = 0x06
    # ADDR_BYTE = 0x07
    CHECK_BYTE = 0x07
    DEVICE_COMM = 0x6C
    ROW_EN = int(2 ** 4 - 1)
    # ROW_EN = int(2 ** 1)
    # 007 716 223 331 // 130(wlen=013) 726(wlen=1) 723/721 wlen=1
    WL_CTRL = 2
    COL_EN = int(2 ** 3)
    BL_CTRL = 3
    print(bin(ROW_EN), bin(COL_EN), bin(WL_CTRL), bin(BL_CTRL))
    # return
    data_EN = 0
    crossbar_ADDR = 0
    Symbol = True
    readin = array('B', [0x01])
    while True:
        Input = np.random.randint(0, 16, 4).tolist()
        Input = [15, 0, 0, 0]
        break
        TempSum = Input[0] + Input[1] + Input[2] + Input[3]
        if 15 < TempSum < 20:
            break
    print(Input)
    dataInA = Input[0] << 4
    dataInA += Input[1]
    dataInB = Input[2] << 4
    dataInB += Input[3]
    dataInA = array('B', [0x09, dataInA])
    dataInB = array('B', [0x0a, dataInB])
    # dataInA = array('B', [0x09, 0xff])
    # dataInB = array('B', [0x0a, 0x0f])
    data_EN += ROW_EN
    data_EN <<= 4
    data_EN += COL_EN
    crossbar_ADDR += WL_CTRL
    crossbar_ADDR <<= 3
    crossbar_ADDR += BL_CTRL
    crossbar_ADDR <<= 1
    crossbar_ADDR += Symbol
    NotSuccess = GroupConfig(data_EN=data_EN, data_ADDR=crossbar_ADDR, handle=handle, DEVICE_COMM=DEVICE_COMM)
    data_EN = array('B', [ENABLE_BYTE, data_EN])
    
    # print(bin(crossbar_ADDR))
    crossbar_ADDR = array('B', [ADDR_BYTE, crossbar_ADDR])
    # # Write in the value of enabled part of the crossbar
    NotSuccess = Communication_Write(handle=handle, data=dataInA, DEVICE=DEVICE_COMM, mode=AA_I2C_NO_FLAGS)
    NotSuccess = Communication_Write(handle=handle, data=dataInB, DEVICE=DEVICE_COMM, mode=AA_I2C_NO_FLAGS)
    NotSuccess = Communication_Write(handle=handle, data=data_EN, DEVICE=DEVICE_COMM, mode=AA_I2C_NO_STOP)
    Communication_Read(handle=handle, data=readin, DEVICE=DEVICE_COMM, mode=AA_I2C_NO_FLAGS)
    print(readin, "**", data_EN)
    NotSuccess = Communication_Write(handle=handle, data=crossbar_ADDR, DEVICE=DEVICE_COMM, mode=AA_I2C_NO_STOP)
    Communication_Read(handle=handle, data=readin, DEVICE=DEVICE_COMM, mode=AA_I2C_NO_FLAGS)
    print(readin, "**", crossbar_ADDR)
    Count_Num = 0
    for i in range(4):
        Count_Num <<= 8
        NotSuccess = Communication_Write(handle=handle, data=array("B", [5 - i]), DEVICE=DEVICE_COMM, mode=AA_I2C_NO_FLAGS)
        Communication_Read(handle=handle, data=readin, DEVICE=DEVICE_COMM, mode=AA_I2C_NO_FLAGS)
        print(readin, "****", bin(readin[0]))
        Count_Num += readin[0]
    print(Count_Num)
    aa_close(handle)
    return

def Communication_Init(mode = AA_CONFIG_GPIO_I2C, pullup = AA_I2C_PULLUP_NONE, bitrate = I2C_BITRATE):
    (num, ports, unique_ids) = aa_find_devices_ext(16, 16)
    port = ports[num - 1]
    handle = aa_open(port)
    aa_configure(handle,  mode)
    aa_i2c_pullup(handle, pullup)
    bitrate = aa_i2c_bitrate(handle, bitrate)
    return handle, bitrate

def Communication_Write(handle, data, DEVICE, mode, Limit = 10):
    TryCount = 0
    result = 0
    while result <= 0:# and TryCount <= Limit:
        result = aa_i2c_write(handle, DEVICE, mode, data)
        TryCount += 1
        # aa_sleep_ms(1)
    return TryCount >= Limit

def Communication_Read(handle, data, DEVICE, mode, Limit = 10):
    TryCount = 0
    result = 0
    while result <= 0:# and TryCount <= Limit:
        result, read_byte = aa_i2c_read(handle, DEVICE, mode, data)
        TryCount += 1
        # aa_sleep_ms(1)
    return TryCount >= Limit

def MultiByteRead(Addr_iterator, handle, DEVICE_COMM):
    X = 0
    readin = array('B', [0x00])
    NotSuccess = False
    for i in Addr_iterator:
        X <<= 8
        data_addr = array('B', [i])
        NotSuccess = Communication_Write(handle=handle, data=data_addr, DEVICE=DEVICE_COMM, mode=AA_I2C_NO_STOP)
        NotSuccess = Communication_Read(handle=handle, data=readin, DEVICE=DEVICE_COMM, mode=AA_I2C_NO_FLAGS)
        if NotSuccess:
            break
        X += int(readin[0])
    return NotSuccess, X

def GroupConfig(data_EN, data_ADDR, handle, DEVICE_COMM, data_WRITE=None, ENABLE_BYTE=0x00, ADDR_BYTE=0x06, WRITE_BYTE=0x08):
    if data_WRITE != None:
        data_WRITE = array("B", [WRITE_BYTE, data_WRITE])
        NotSuccess = Communication_Write(handle=handle, data=data_WRITE, DEVICE=DEVICE_COMM, mode=AA_I2C_NO_FLAGS)
        if NotSuccess:
            return NotSuccess
    data_EN = array("B", [ENABLE_BYTE, data_EN])
    NotSuccess = Communication_Write(handle=handle, data=data_EN, DEVICE=DEVICE_COMM, mode=AA_I2C_NO_FLAGS)
    if NotSuccess:
        return NotSuccess
    data_ADDR = array('B', [ADDR_BYTE, data_ADDR])
    NotSuccess = Communication_Write(handle=handle, data=data_ADDR, DEVICE=DEVICE_COMM, mode=AA_I2C_NO_FLAGS)
    if NotSuccess:
        return NotSuccess
    return NotSuccess

def HandShake(handle, DEVICE_COMM, Initial=0):
    Judge = array('B', [Initial])
    Count = 0
    Limit = 10
    CHECK_BYTE = array('B', [0x07])
    NotSuccess = Communication_Write(handle=handle, data=CHECK_BYTE, DEVICE=DEVICE_COMM, mode=AA_I2C_NO_STOP)
    while int(Judge[0]) == Initial:# and Count < Limit:
        NotSuccess = Communication_Read(handle=handle, data=Judge, DEVICE=DEVICE_COMM, mode=AA_I2C_NO_FLAGS)
        if NotSuccess:
            break
        Count = Count + 1
        # time.sleep(1)
    if NotSuccess or Count >= Limit:
        NotSuccess = True
    return NotSuccess

def Write_Test(COL_NUM=32, COL_BIT=3, ROW_NUM=36, GROUP=4, ENABLE_BYTE=0x00, ADDR_BYTE=0x06, CHECK_BYTE=0x07,
               DEVICE_COMM=0x6C, BreakPoint=False, Target=[3, 7, 0, 3], FileNameA="ResA", FileNameB="ResB"):
    handle, bitrate = Communication_Init()
    print("Bitrate set to %d kHz" % bitrate)
    Symbol = True
    ResStorage = [[] for i in range(ROW_NUM)]
    FileA = open(FileNameA + ".txt", mode="a")
    FileB = open(FileNameB + ".txt", mode="a")
    for ROW_EN in range(GROUP):
        if BreakPoint and ROW_EN < Target[0]:
            continue
        data_EN = int(2 ** ROW_EN)
        data_EN <<= GROUP
        for WL_CTRL in range(int(ROW_NUM / GROUP)):
            if BreakPoint and WL_CTRL < Target[1]:
                continue
            crossbar_ADDR = WL_CTRL
            crossbar_ADDR <<= COL_BIT
            for COL_EN in range(GROUP):
                if BreakPoint and COL_EN < Target[2]:
                    continue
                data_EN += int(2 ** COL_EN)
                for BL_CTRL in range(int(COL_NUM / GROUP)):
                    if BreakPoint and BL_CTRL < Target[3]:
                        continue
                    BreakPoint = False
                    crossbar_ADDR += BL_CTRL
                    crossbar_ADDR <<= 1
                    crossbar_ADDR += Symbol
                    print(bin(crossbar_ADDR), "****")
                    NotSuccess = GroupConfig(data_EN=data_EN, data_ADDR=crossbar_ADDR, handle=handle, DEVICE_COMM=DEVICE_COMM,
                                             ENABLE_BYTE=ENABLE_BYTE, ADDR_BYTE=ADDR_BYTE)
                    if NotSuccess:
                        break
                    data_addr = array('B', [CHECK_BYTE])
                    NotSuccess = Communication_Write(handle=handle, data=data_addr, DEVICE=DEVICE_COMM, mode=AA_I2C_NO_STOP)
                    if NotSuccess:
                        break
                    NotSuccess = HandShake(handle=handle, DEVICE_COMM=DEVICE_COMM)
                    if NotSuccess:
                        break
                    NotSuccess, ResA = MultiByteRead(range(5, 1, -1), handle, DEVICE_COMM)
                    if NotSuccess:
                        break
                    NotSuccess, ResB = MultiByteRead(range(11, 7, -1), handle, DEVICE_COMM)
                    if NotSuccess:
                        break
                    ResStorage.append((int(ResA), int(ResB)))
                    print("ROW_EN =", ROW_EN, "WL_CTRL =", WL_CTRL,
                          "COL_EN =", COL_EN, "BL_CTRL =", BL_CTRL)
                    print(ResA, ResB, NotSuccess)
                    if NotSuccess or ResA > MAX_LIMIT or ResB > MAX_LIMIT:
                        break
                    crossbar_ADDR = crossbar_ADDR - int(Symbol)
                    crossbar_ADDR >>= 1
                    crossbar_ADDR -= BL_CTRL
                    Symbol += 1
                    Symbol %= 2
                    Symbol = bool(Symbol)
                    ResA = str(ResA)
                    ResB = str(ResB)
                    print(f"{ResA:8}", file=FileA, end=' ')
                    print(f"{ResB:8}", file=FileB, end=' ')
                data_EN -= int(2 ** COL_EN)
            print(file=FileA)
            print(file=FileB)
    # print(ResStorage, file=File)
    if NotSuccess:
        print("Data Address write failed when writing Row " + str(ROW_EN * (ROW_NUM / GROUP) + WL_CTRL) +
                " Col " + str(COL_EN * (COL_NUM / GROUP) + BL_CTRL))
    FileA.close()
    FileB.close()
    aa_close(handle)
    return

def FindMatchGroup(FileNameA="ResA", FileNameB="ResB", GROUP=4, ROW_NUM=36, COL_NUM=32, STANDARD=4000):
    FileA = open(FileNameA + ".txt", mode="r")
    FileB = open(FileNameB + ".txt", mode="r")
    ResA = FileA.read().split('\n')[:-1]
    ResB = FileB.read().split('\n')[:-1]
    Diff = []
    for i in range(len(ResA)):
        ResA[i] = ResA[i].split(' ')[:-1]
        ResB[i] = ResB[i].split(' ')[:-1]
        Diff.append([])
        ResA[i][:] = filterfalse(lambda x : x == '', ResA[i])
        ResB[i][:] = filterfalse(lambda x : x == '', ResB[i])
    for i in range(len(ResA)):
        for j in range(len(ResA[i])):
            ResA[i][j] = int(ResA[i][j])
            ResB[i][j] = int(ResB[i][j])
            Diff[i].append(abs(ResA[i][j] - ResB[i][j]))
    STEPR = int(ROW_NUM / GROUP)
    STEPC = int(COL_NUM / GROUP)
    WL_IND = []
    BL_IND = []
    COL_IND = []
    for WL_CTRL in range(STEPR):
        for COL_EN in range(GROUP):
            for BL_CTRL in range(STEPC):
                CHECK = 0
                for i in range(GROUP):
                    CHECK += int(Diff[WL_CTRL + STEPR * i][BL_CTRL + COL_EN * STEPC] > STANDARD)
                if CHECK == GROUP:
                    print("Possible Group: WL_CTRL =", WL_CTRL, "BL_CTRL =", BL_CTRL, "COL_EN =", COL_EN)
                    WL_IND.append(WL_CTRL)
                    BL_IND.append(BL_CTRL)
                    COL_IND.append(COL_EN)
    print("Total Number of available device group:", len(WL_IND))
    print("Yield is:", len(WL_IND) / COL_NUM / ROW_NUM * GROUP)
    FileA.close()
    FileB.close()
    return WL_IND, BL_IND, COL_IND

def Write_Target(WL, BL, COL, WeightList, Break = [0, 0]):
    COL_BIT = 3
    GROUP = 4
    DEVICE_COMM = 0x6C
    STANDARD = 60000
    handle, bitrate = Communication_Init()
    print("Bitrate set to %d kHz" % bitrate)
    Symbol = True
    BreakPoint = Break
    FailList = []
    FailChannel = []
    File = open("FailList.txt", mode="a")
    Flag = True
    for i in range(int(BreakPoint[0] / GROUP), int(len(WeightList) / GROUP)):
        for ROW_EN in range(GROUP):
            if ROW_EN < BreakPoint[1] and Flag:
                continue
            Flag = False
            data_EN = int(2 ** ROW_EN)
            data_EN <<= GROUP
            data_EN += int(2 ** COL[i])
            data_ADDR = WL[i]
            data_ADDR <<= COL_BIT
            data_ADDR += BL[i]
            data_ADDR <<= 1
            data_ADDR += int(Symbol)
            NotSuccess = GroupConfig(data_EN=data_EN, data_ADDR=data_ADDR, data_WRITE=int(WeightList[GROUP * i + ROW_EN]), handle=handle, DEVICE_COMM=DEVICE_COMM)
            if NotSuccess:
                BreakPoint = [i, ROW_EN]
                print("Break Point number:", i, "WL_EN :", ROW_EN, "WL_CTRL:", WL[i], "COL_EN", COL[i], "BL_CTRL", BL[i], "Weight", WeightList[i])
                break
            Symbol += 1
            Symbol %= 2
            Symbol = bool(Symbol)
            NotSuccess = HandShake(handle=handle, DEVICE_COMM=DEVICE_COMM)
            if NotSuccess:
                NotSuccess = True
                BreakPoint = [i, ROW_EN]
                print("Break Point number:", i, "WL_EN :", ROW_EN, "WL_CTRL:", WL[i], "COL_EN", COL[i], "BL_CTRL", BL[i], "Weight", WeightList[i])
                break
            NotSuccess, Verify_Value = MultiByteRead(range(5, 1, -1), handle, DEVICE_COMM)
            if int(Verify_Value < STANDARD) == int(WeightList[i]):
                continue
            else:
                FailList.append([ROW_EN, WL[i], COL[i], BL[i], WeightList[GROUP * i + ROW_EN]])
                print(ROW_EN, WL[i], COL[i], BL[i], WeightList[GROUP * i + ROW_EN], file=File)
                FailChannel.append(i)
        if NotSuccess:
            print("Break Point number:", i, "WL_EN :", ROW_EN, "WL_CTRL:", WL[i], "COL_EN", COL[i], "BL_CTRL", BL[i], "Weight", WeightList[i])
            break
    File.close()
    aa_close(handle)
    return FailList, FailChannel

def Write_Fail(FailList=None):
    if FailList == None:
        FailFile = open("FailList.txt", mode="r")
        FailList = FailFile.read().split("\n")[:-1]
        for String in FailList:
            String = String.split(' ')
            for item in String:
                item = int(item)
        FailFile.close()
    COL_BIT = 3
    GROUP = 4
    DEVICE_COMM = 0x6C
    STANDARD = 60000
    Symbol = True
    handle, bitrate = Communication_Init()
    BreakPoint = 0
    print("Bitrate set to %d kHz" % bitrate)
    File = open("FailList.txt", mode="w")
    for i in range(len(FailList)):
        ROW_EN, WL, COL_EN, BL, Target_Weight = FailList[i]
        data_EN = int(2 ** ROW_EN)
        data_EN <<= GROUP
        data_EN += int(2 ** COL_EN)
        data_ADDR = WL
        data_ADDR <<= COL_BIT
        data_ADDR += BL
        data_ADDR <<= 1
        data_ADDR += int(Symbol)
        NotSuccess = GroupConfig(data_EN=data_EN, data_ADDR=data_ADDR, data_WRITE=int(Target_Weight), handle=handle, DEVICE_COMM=DEVICE_COMM)
        if NotSuccess:
            BreakPoint = i
            break
        Symbol += 1
        Symbol %= 2
        Symbol = bool(Symbol)
        NotSuccess = HandShake(handle=handle, DEVICE_COMM=DEVICE_COMM)
        if NotSuccess:
            NotSuccess = True
            BreakPoint = i
            break
        NotSuccess, Verify_Value = MultiByteRead(range(5, 1, -1), handle, DEVICE_COMM)
        if int(Verify_Value < STANDARD) == Target_Weight:
            continue
        else:
            FailList.append([ROW_EN, WL, COL_EN, BL, Target_Weight])
            print(ROW_EN, WL, COL_EN, BL, Target_Weight, file=File)
    aa_close(handle)
    return FailList

def HardwareApply(Input, Model, HardwareResult):
    TempResult = Model.Decoder.FC_hidden(Input)
    for i in range(batch_size):
        for j in range(min(hidden_dim3, len(HardwareResult[i]))):
            HResult = Round.apply(HardwareResult[i][j] / (Model.Decoder.FC_hidden.max - Model.Decoder.FC_hidden.min) * 16)\
                                    / 16 * (Model.Decoder.FC_hidden.max - Model.Decoder.FC_hidden.min)
            # print(HResult, TempResult[i, j])
            TempResult[i, j] = HResult
    Result = Model.Decoder.LeakyReLU(TempResult)
    Result = Model.Decoder.LeakyReLU(Model.Decoder.FC_hidden2(Result))
    Result = Model.Decoder.LeakyReLU(Model.Decoder.FC_hidden3(Result))
    Result = torch.sigmoid(Model.Decoder.FC_output(Result))
    return Result

def GetScaleOut(Raw):
    x = 0
    if Raw < 85000:
        x += 1
    if Raw < 42000:
        x += 1
    if x < 34000:
        x += 1
    if x < 22000:
        x += 1
    return x

def Parallel_Compute(BatchData, AvailablePlace, WeightList):
    WL, BL, COL = AvailablePlace
    COL_BIT = 3
    GROUP = 4
    DEVICE_COMM = 0x6C
    COMPUTE_BYTE = 0x09
    Symbol = 1
    handle, bitrate = Communication_Init()
    result_addr = array('B', [0x02])
    Result = 0
    ChannelList = []
    print("Bitrate set to %d kHz" % bitrate)
    for i in range(int(len(WL) / GROUP)):
        data_EN = int(2 ** GROUP - 1)
        data_EN <<= GROUP
        data_EN += int (2 ** COL[i])
        data_ADDR = WL[i]
        data_ADDR <<= COL_BIT
        data_ADDR += BL[i]
        data_ADDR <<= 1
        data_ADDR += Symbol
        data_COMP = 0
        print("WL:", WL[i], "COL_EN:", COL[i], "BL:", BL[i])
        store_EN = int(2 ** 1)
        store_EN <<= 4
        store_EN += int(2 ** 2)
        store_ADDR = 7
        store_ADDR <<= 3
        store_ADDR += 3
        store_ADDR <<= 1
        store_ADDR += Symbol
        NotSuccess = GroupConfig(data_EN=store_EN, data_ADDR=store_ADDR, handle=handle, DEVICE_COMM=DEVICE_COMM,
                                    ENABLE_BYTE=0x01, ADDR_BYTE=0x07)
        if NotSuccess:
            break
        if False:
            for j in range(GROUP):
                data_COMP = BatchData[(i * GROUP + j) % len(BatchData)]
                Result += data_COMP * WeightList[i * GROUP + j]
        else:
            for j in range(GROUP):
                data_COMP <<= GROUP
                data_COMP += int(BatchData[(i * GROUP + GROUP - j - 1) % len(BatchData)] * WeightList[i * GROUP + j])
                print(j, "Weight:", WeightList[i * GROUP + j], "Input:", int(BatchData[(i * GROUP + GROUP - j - 1) % len(BatchData)]))
                if j % 2 == 1:
                    data_COMP = array('B', [COMPUTE_BYTE + int(j / 2), abs(data_COMP)])
                    NotSuccess = Communication_Write(handle=handle, data=data_COMP, DEVICE=DEVICE_COMM, mode=AA_I2C_NO_FLAGS)
                    if NotSuccess:
                        break
                    data_COMP = 0
            if NotSuccess:
                break
            NotSuccess = GroupConfig(data_EN=data_EN, data_ADDR=data_ADDR, handle=handle, DEVICE_COMM=DEVICE_COMM)
            if NotSuccess:
                break
            Symbol += 1
            Symbol %= 2
            NotSuccess = HandShake(handle=handle, DEVICE_COMM=DEVICE_COMM)
            if NotSuccess:
                break
            readin = array('B', [0])
            NotSuccess = Communication_Write(handle=handle, data=result_addr, DEVICE=DEVICE_COMM, mode=AA_I2C_NO_STOP)
            NotSuccess = Communication_Read(handle=handle, data=readin, DEVICE=DEVICE_COMM, mode=AA_I2C_NO_FLAGS)
            # NotSuccess, RawResult = MultiByteRead(range(5, 1, -1), handle, DEVICE_COMM)
            # if RawResult > 200000:
            #     NotSuccess = False
            #     return None
            # ScaleOut = GetScaleOut(RawResult)
            # print("Raw Output:", RawResult, "Scale Output:", ScaleOut)
            Result += readin[0]
        if i % GROUP == 3:
            ChannelList.append(Result)
            Result = 0
    Hardware_Result = []
    for i in range(int(len(ChannelList) / 2)):
        Hardware_Result.append(ChannelList[2 * i] - ChannelList[2 * i + 1])
    aa_close(handle)
    # print(BatchData)
    # print(WeightList)
    # print(Hardware_Result)
    return Hardware_Result

def Write(handle, Value, Symbol, Enable, DEVICE_COMM, Write_Addr, MODE_BYTE, CHECK_BYTE, ADDR_BYTE, ENABLE_BYTE, MODE = [1, 0]):
    # Select Mode to the Correct Mode
    Write_Message = array('B', [MODE_BYTE, MODE[int(Value)]])
    NotSuccess = Communication_Write(handle=handle, data=Write_Message, DEVICE=DEVICE_COMM, mode=AA_I2C_NO_FLAGS)
    # Doing
    Write_Addr <<= 1
    Write_Addr += Symbol
    NotSuccess = GroupConfig(Enable, Write_Addr, handle, DEVICE_COMM, ENABLE_BYTE=ENABLE_BYTE, ADDR_BYTE=ADDR_BYTE)
    return NotSuccess

def Refresh(WL_SUB, COL_BIT, BL_SUB, handle, EnableSub, Symbol, weight_addr):
    DEVICE_COMM = 0x6C
    ENABLE_BYTE_SUB = 0x01
    ADDR_BYTE = 0x06
    ADDR_BYTE_SUB = 0x07
    MODE_BYTE = 0x02
    CHECK_BYTE = 0x07
    weight_addr_sub = WL_SUB
    weight_addr_sub <<= COL_BIT
    weight_addr_sub += BL_SUB
    NotSuccess= Write(handle, 1, Symbol, EnableSub, DEVICE_COMM, weight_addr_sub,
                                MODE_BYTE, CHECK_BYTE, ADDR_BYTE_SUB, ENABLE_BYTE_SUB, MODE=[5, 4])
    weight_addr <<= 1
    weight_addr += Symbol
    Write_Message = array('B', [ADDR_BYTE, weight_addr])
    NotSuccess = Communication_Write(handle=handle, data=Write_Message, DEVICE=DEVICE_COMM, mode=AA_I2C_NO_FLAGS)
    weight_addr -= Symbol
    weight_addr >>= 1
    Symbol = (Symbol + 1) % 2
    HandShake(handle, DEVICE_COMM, Symbol)
    NotSuccess= Write(handle, 0, Symbol, EnableSub, DEVICE_COMM, weight_addr_sub,
                                MODE_BYTE, CHECK_BYTE, ADDR_BYTE_SUB, ENABLE_BYTE_SUB, MODE=[5, 4])
    weight_addr <<= 1
    weight_addr += Symbol
    Write_Message = array('B', [ADDR_BYTE, weight_addr])
    NotSuccess = Communication_Write(handle=handle, data=Write_Message, DEVICE=DEVICE_COMM, mode=AA_I2C_NO_FLAGS)
    weight_addr -= Symbol
    weight_addr >>= 1
    Symbol = (Symbol + 1) % 2
    HandShake(handle, DEVICE_COMM, Symbol)
    return Symbol

def Parallel_Compute_Serial_Ctrl(Data, AvailablePlace, Weight, model, StorePlace, Initial_Weight=[1, 1, 1, 1], GROUP_TOTAL=4, Result_Dim=hidden_dim3):
    WL, BL, COL = AvailablePlace
    WL_SUB, BL_SUB, COL_SUB, ROW_SUB = StorePlace
    COL_BIT = 3
    DATA_BIT = 4
    GROUP = 4
    Parallelism = 4
    InputChannel = 16
    DEVICE_COMM = 0x6C
    ENABLE_BYTE = 0x00
    ENABLE_BYTE_SUB = 0x01
    ADDR_BYTE = 0x06
    ADDR_BYTE_SUB = 0x07
    COMPUTE_BYTE = 0x09
    MODE_BYTE = 0x02
    COMPUTE_MODE = 0x02
    STORE_READ_MODE = 0x03
    RESULT_BYTE = 0x02
    CHECK_BYTE = 0x07
    weight_addr = WL
    weight_addr <<= COL_BIT
    weight_addr += BL
    Symbol = 1
    handle, bitrate = Communication_Init()
    result_addr = array('B', [RESULT_BYTE])
    print("Bitrate set to %d kHz" % bitrate)
    # Data Quantization
    Data = torch.tensor(Data, device=DEVICE)
    Data = torch.clip(Round.apply((Data - model.Decoder.FC_hidden.InMin) 
                                  / (model.Decoder.FC_hidden.InMax - model.Decoder.FC_hidden.InMin) * 15), 0, 15).numpy().astype(int)
    # print(Data.shape)
    Result_Batch = np.zeros((Data.shape[0], Result_Dim))
    Test_Result = 0
    Overall_Test = 0
    Sign = 1
    ratio = 0
    monument = 0.1
    count = 0
    for i in range(int(len(Weight) / Parallelism)):
        data_LIST = []
        data_TEMP = 0
        # Write In Weight
        ratio = i / (int(len(Weight) / Parallelism))
        if ratio > monument:
            monument = min(ratio + 0.1, 1)
            print(i, '/', int(len(Weight) / Parallelism))
            # Symbol = Refresh(WL_SUB, COL_BIT, BL_SUB, handle, EnableSub, Symbol, weight_addr)
        if i % 300 == 0:
            time.sleep(1)
        for j in range(Parallelism):
            if Weight[Parallelism * i + j] == Initial_Weight[j]:
                continue
            else:
                # Enable = int(2 ** j)
                # Enable <<= GROUP
                # Enable += int(2 ** COL)
                # NotSuccess= Write(handle, Weight[Parallelism * i + j], Symbol, Enable,
                #                           DEVICE_COMM, weight_addr, MODE_BYTE, CHECK_BYTE, ADDR_BYTE, ENABLE_BYTE)
                # Write_Message = array('B', [CHECK_BYTE])
                # Symbol = (Symbol + 1) % 2
                # HandShake(handle, DEVICE_COMM, Symbol)
                Initial_Weight[j] = Weight[Parallelism * i + j]
                # print("Write Weight", j, "as Value", Weight[Parallelism * i + j])
                # time.sleep(1)
                # os.system("pause")
        Index = i % GROUP_TOTAL
        Enable = int(2 ** GROUP - 1)
        Enable <<= GROUP
        Enable += int(2 ** COL)
        readin = array('B', [0])
        EnableSub = int(2 ** ROW_SUB)
        EnableSub <<= GROUP
        EnableSub += int(2 ** COL_SUB)
        weight_addr_sub = WL_SUB
        weight_addr_sub <<= COL_BIT
        weight_addr_sub += BL_SUB
        Zero = True
        for j in range(Data.shape[0]):
            # Write In Data
            data_LIST = []
            for k in range(Parallelism):
                data_TEMP <<= DATA_BIT
                Test_Result += Data[j, Index * Parallelism + k] * Weight[Parallelism * i + k]
                # print(Test_Result, Data[j, Index * Parallelism + k] * Weight[Parallelism * i + k])
                data_TEMP += int(Data[j, Index * Parallelism + k] * Weight[Parallelism * i + k])
                if k % 2 == 1:
                    data_LIST.append(data_TEMP)
                    if Test_Result > 15:
                        Zero = False
                    data_TEMP = 0
            if Zero:
                continue
            Zero = True
            for k in range(len(data_LIST)):
                data_COMP = array('B', [COMPUTE_BYTE + int(k / 2), abs(data_LIST[k])])
                NotSuccess = Communication_Write(handle=handle, data=data_COMP, DEVICE=DEVICE_COMM, mode=AA_I2C_NO_FLAGS)
            # print(Test_Result, file=fileA, end=' ')
            # Initialize Buffer
            NotSuccess= Write(handle, 1, Symbol, EnableSub, DEVICE_COMM, weight_addr_sub,
                                      MODE_BYTE, CHECK_BYTE, ADDR_BYTE_SUB, ENABLE_BYTE_SUB, MODE=[5, 4])
            # Trigger Action
            weight_addr <<= 1
            weight_addr += Symbol
            Write_Message = array('B', [ADDR_BYTE, weight_addr])
            NotSuccess = Communication_Write(handle=handle, data=Write_Message, DEVICE=DEVICE_COMM, mode=AA_I2C_NO_FLAGS)
            weight_addr -= Symbol
            weight_addr >>= 1
            Symbol = (Symbol + 1) % 2
            HandShake(handle, DEVICE_COMM, Symbol)
            # print("Set Buffer RRAM")
            # time.sleep(1)
            # os.system("pause")
            # Select Compute Mode
            Write_Message = array('B', [MODE_BYTE, COMPUTE_MODE])
            NotSuccess = Communication_Write(handle=handle, data=Write_Message, DEVICE=DEVICE_COMM, mode=AA_I2C_NO_FLAGS)
            # Inference & Stored in Buffer
            weight_addr <<= 1
            weight_addr += Symbol
            NotSuccess = GroupConfig(data_EN=Enable, data_ADDR=weight_addr, handle=handle, DEVICE_COMM=DEVICE_COMM)
            weight_addr -= Symbol
            weight_addr >>= 1
            Symbol = (Symbol + 1) % 2
            HandShake(handle, DEVICE_COMM, Initial=Symbol)
            # print("Computation Done")
            # time.sleep(1)
            # os.system("pause")
            # Select Buffer Read Mode
            Write_Message = array('B', [MODE_BYTE, STORE_READ_MODE])
            NotSuccess = Communication_Write(handle=handle, data=Write_Message, DEVICE=DEVICE_COMM, mode=AA_I2C_NO_FLAGS)
            # Trigger Read
            weight_addr_sub <<= 1
            weight_addr_sub += Symbol
            Write_Message = array('B', [ADDR_BYTE_SUB, weight_addr_sub])
            NotSuccess = Communication_Write(handle=handle, data=Write_Message, DEVICE=DEVICE_COMM, mode=AA_I2C_NO_FLAGS)
            weight_addr_sub -= Symbol
            weight_addr_sub >>= 1

            weight_addr <<= 1
            weight_addr += Symbol
            Write_Message = array('B', [ADDR_BYTE, weight_addr])
            NotSuccess = Communication_Write(handle=handle, data=Write_Message, DEVICE=DEVICE_COMM, mode=AA_I2C_NO_FLAGS)
            weight_addr -= Symbol
            weight_addr >>= 1
            Symbol = (Symbol + 1) % 2
            HandShake(handle, DEVICE_COMM, Initial=Symbol)
            # print("Read Out Result")
            # time.sleep(1)
            # os.system("pause")
            # Read Out Result
            NotSuccess = Communication_Write(handle=handle, data=result_addr, DEVICE=DEVICE_COMM, mode=AA_I2C_NO_STOP)
            NotSuccess = Communication_Read(handle=handle, data=readin, DEVICE=DEVICE_COMM, mode=AA_I2C_NO_FLAGS)
            # Result_Batch[j, int(i / GROUP_TOTAL / 2)] += Sign * Test_Result / 15
            Result_Batch[j, int(i / GROUP_TOTAL / 2)] += Sign * min(3, readin[0])
            # print(Test_Result, min(3, readin[0]), file=fileA)
            # print(Sign * readin[0], int(Test_Result / 15))
            # Overall_Test += Sign * round(Test_Result / 15)
            Test_Result = 0
        if i % GROUP_TOTAL == GROUP_TOTAL - 1:
            Sign = -Sign
            # print(Result_Batch[j, int(i / GROUP / 2)], Overall_Test)
            # return
        # print(file=fileA)
    aa_close(handle)
    # print(Result_Batch.shape)
    return Result_Batch

def DeployWeightLog(DeployWeight):
    with open("DeployWeight.csv", "w", encoding='utf-8', newline='') as f:
        csv_writer = csv.writer(f)
        for item in DeployWeight:
            csv_writer.writerow(item)
        f.close()
    return

def VAE_DEMO():
    # ROW_NUM = 36
    # COL_NUM = 32
    # COL_BIT = 3
    # GROUP = 4
    # Write_Test()
    # WL_LIST, BL_LIST, COL_LIST = FindMatchGroup()
    # Write_Test(ENABLE_BYTE=0x01, ADDR_BYTE=0x07, FileNameA="ResC", FileNameB="ResD")
    # WL_LIST_SUB, BL_LIST_SUB, _ = FindMatchGroup(FileNameA="ResC", FileNameB="ResD", GROUP=1)
    # WL_LIST_SUB = [j for i in WL_LIST_SUB for j in i]
    # BL_LIST_SUB = [j for i in BL_LIST_SUB for j in i]
    # StoreList = []
    # for WL_ID in WL_LIST:
    #     for BL_ID in BL_LIST:
    #         data_EN = int(WL_ID / ROW_NUM * GROUP)
    #         data_EN <<= GROUP
    #         data_EN += int(BL_ID / COL_NUM * GROUP)
    #         data_ADDR = int(WL_ID % (ROW_NUM / GROUP))
    #         data_ADDR <<= COL_BIT
    #         data_ADDR += int(BL_ID % (COL_NUM / GROUP))
    #         StoreList.append([data_EN, data_ADDR])
    # BUFFER_LIST = [[] for i in range(GROUP)]
    # for LIST in StoreList:
    #     BUFFER_LIST[LIST[2]].append(LIST)
    # GROUP_LEN = min([len(LIST) for LIST in BUFFER_LIST])
    # StoreList = []
    # for i in range(GROUP_LEN):
    #     for j in range(GROUP):
    #         StoreList.append(BUFFER_LIST[j][i])
    # print(StoreList)
    model = torch.load('C:\\Users\\16432\\Desktop\\Workplace\\Analog_Memory_DEMO\\VAE_DEMO\\checkpoint_test_with_hardware.ptr', map_location=DEVICE)
    # for name, param in model.named_parameters():
    #     print(name)
    # encoder = Encoder(Input_Dim=x_dim, Hidden_Dim=hidden_dim, Hidden_Dim2=hidden_dim2,
    #                   Hidden_Dim3=hidden_dim3, Latent_Dim=latent_dim)
    # decoder = Decoder(Latent_Dim=latent_dim, Hidden_Dim=hidden_dim, Hidden_Dim2=hidden_dim2,
    #                   Hidden_Dim3=hidden_dim3, Output_Dim=x_dim)
    # model = Model(ENCODER=encoder, DECODER=decoder).to(DEVICE)
    # print(model.Decoder.FC_hidden.min)
    weight = Sign.apply(model.Decoder.FC_hidden.layer.weight).detach().numpy().transpose()
    DeployList = np.zeros((weight.shape[0], 2 * weight.shape[1]))
    weight = weight.transpose()
    # print(weight.shape)
    for i in range(weight.shape[1]):
        for j in range(weight.shape[0]):
            if weight[j, i] == 1:
                DeployList[i][2 * j] = 1
            elif weight[j, i] == -1:
                DeployList[i][2 * j + 1] = 1
    print(DeployList.shape)
    DeployWeightLog(DeployList)
    # return
    Parallelism = 4
    WeightList = []
    DataChannel = DeployList.shape[0]
    DeployChannel = DeployList.shape[1]
    # print(DeployList)
    for i in range(DeployChannel):
        for j in range(DataChannel):
            WeightList.append(DeployList[j][i])
    # print(WeightList)
    # return
    # os.system("pause")
    # FailList, FailChannel = [], []
    # FailList, FailChannel = Write_Target(WL_LIST, BL_LIST, COL_LIST, WeightList=WeightList)
    # print("First Write Loop End, Fail List length is", len(FailList))
    # os.system("pause")
    # Count = 0
    # Limit = 10
    # while len(FailList) > 0 and Count < Limit:
    #     FailList = Write_Fail(FailList)
    #     Count += 1
    #     os.system("pause")
    # if Count >= Limit:
    #     print("Write Failed, Fail List Length is ", len(FailList))
    #     return
    model.eval()
    count = 0
    overall_loss = 0
    overall_loss_origin = 0
    BatchData = []
    InputList = []
    MeanList = []
    VarList = []
    with torch.no_grad():
        for batch_idx, (x, label) in enumerate(tqdm(test_loader)):
            # if batch_idx < 129 or label == torch.tensor([0]) or label == torch.tensor([1])\
            #                   or label == torch.tensor([2]) or label == torch.tensor([3])\
            #                   or label == torch.tensor([4]) or label == torch.tensor([5])\
            #                   or label == torch.tensor([6]) or label == torch.tensor([7])\
            #                   or label == torch.tensor([7]) or label == torch.tensor([9]):
            #     continue
            print(batch_idx, label)
            x = x.view(batch_size, x_dim)
            x = x.to(DEVICE)
            # x = torch.round(x * 4) / 4
            Mean, Log_Var = model.Encoder(x)
            z = reparameterization(Mean, torch.exp(0.5 * Log_Var))  # takes exponential function (log var -> var)
            BatchData.append(z.numpy())
            x_hat_active = model.Decoder(z)
            loss = loss_function(x, x_hat_active, Mean, Log_Var)
            InputList.append(x)
            MeanList.append(Mean)
            VarList.append(Log_Var)
            overall_loss_origin += loss
            count += 1
            break
    BatchData = np.vstack(BatchData)
    BatchNum = BatchData.shape[0] / count
    # print(BatchNum)
    # print(count, BatchData.shape, overall_loss_origin / BatchData.shape[0])
    # show_image(x, idx=0)
    # show_image(x_hat_active, idx=0)
    # return
    # print(model.Decoder.FC_hidden.scale) 124.0069 v.s. 176.7617
    # return
    # save_image(x, "reference.png")
    # save_image(x_hat, "test.png")
    # show_image(x, idx=0)
    # show_image(x_hat, idx=0)
    #
    # # Plot Generated Picture from Noise
    AvailablePlace = [2, 3, 2]
    StorePlace = [7, 3, 2, 1]
    with torch.no_grad():
        overall_loss = 0
        HardwareResult = Parallel_Compute_Serial_Ctrl(BatchData, AvailablePlace, WeightList, model, StorePlace)
        # print(HardwareResult)
        for i in range(count):
            Input = torch.tensor(HardwareResult[int(i * batch_size): int((i + 1) * batch_size)], device=DEVICE).reshape((batch_size, hidden_dim3))
            Min = torch.ones((batch_size, latent_dim)) * model.Decoder.FC_hidden.InMin
            bias = (Min / (model.Decoder.FC_hidden.InMax - model.Decoder.FC_hidden.InMin)).detach()
            w = Sign.apply(model.Decoder.FC_hidden.layer.weight)
            x = x * (model.Decoder.FC_hidden.InMax - model.Decoder.FC_hidden.InMin)
            x = Input + torch.nn.functional.linear(bias, w)
            h = model.Decoder.LeakyReLU(x).type(torch.float)
            h = model.Decoder.LeakyReLU(model.Decoder.FC_hidden2(h))
            h = model.Decoder.LeakyReLU(model.Decoder.FC_hidden3(h))
            x_hat = torch.sigmoid(model.Decoder.FC_output(h))
            loss = loss_function(InputList[i], x_hat, MeanList[i], VarList[i])
            overall_loss += loss.item()
    # print(overall_loss / count / batch_size)
    # File = open("Test.txt", mode='w')
    # for list in DeployList:
    #     for item in list:
    #         print(item, end=' ', file=File)
    #     print(file=File)
    # save_image(origin_generated_images.view(batch_size, 1, 28, 28), 'origin_generated_sample.png')
    # save_image(generated_images.view(batch_size, 1, 28, 28), 'generated_sample.png')
    num = 50
    x_hat = torch.cat([x_hat_active.view(28, 28), x_hat.view(28, 28)], 1)
    print(x_hat.size())
    show_image(x_hat, idx=0)
    point = overall_loss_origin.item() / count / batch_size
    point = '%.4f' % point
    # print("original point = ", overall_loss_origin / count / batch_size)
    # save_image(x_hat_active.view(batch_size, 1, 28, 28), 'C:\\Users\\16432\\Desktop\\Workplace\\Python\\VAE_DEMO\\data_fig\\origin_' + str(num) + '_' + point + '.png')
    point = overall_loss / count / batch_size
    point = '%.4f' % point
    # print("deployed point = ", overall_loss / count / batch_size)
    # save_image(x_hat.view(batch_size, 1, 28, 28), 'C:\\Users\\16432\\Desktop\\Workplace\\Python\\VAE_DEMO\\data_fig\\deployed_' + str(num) + '_' + point + '.png')
    # show_image(generated_images, idx=0)
    # show_image(generated_images, idx=1)
    # show_image(generated_images, idx=10)
    # show_image(generated_images, idx=20)
    # show_image(generated_images, idx=50)
    return

def CNN_DEMO():
    folderOut = './Analog_Memory_DEMO/cnn_fixpoint/1D_CNN_dataset'
    pat_path = folderOut + '/chb' + '02'
    mean_se = np.load(pat_path + "/mean_stat.npy")
    std_se = np.load(pat_path + "/std_stat.npy")
    Parallellism = 4

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    folderIn = folderOut + '/chb' + '02' + '/test' + str(0)
    test_dataset = EEGDataset(folderIn, train=False, mean=mean_se, std=std_se)
    # test_loader = DataLoader(dataset=test_dataset, batch_size=len(test_dataset), shuffle=False)
    batch_size = 32
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    
    model = Net().to(device)
    model.quantize(1, 2, 2, 4)
    model.load_state_dict(torch.load(('./Analog_Memory_DEMO/cnn_fixpoint/ckpt/1d_cnn_2bit_sdj_avgpool.pt'), map_location='cpu'))
    count = 0
    pos = 0
    neg = 0
    pos_lim = 5
    neg_lim = 0
    TN = 0
    TP = 0
    FP = 0
    FN = 0

    TN_O = 0
    TP_O = 0
    FP_O = 0
    FN_O = 0
    log = []
    for i, (data, target) in enumerate(test_loader, 1):
        if count < 2400:
            # pos += torch.sum(target)
            # neg += batch_size - pos
            count += batch_size
            continue
        print("Begin Number", count, pos, neg)
        data, target = data.to(device), target.to(device)
        x = model.qconv1.fake_i.apply(data, model.qconv1.qi)
        q_data = x.clone()
        x_int = x * 8 + 6
        Input_Bias = 6 * torch.ones(x.size(), device=DEVICE)
        q_w = FakeQuantize.apply(model.qconv1.conv_module.weight, model.qconv1.qw)
        q_int = q_w * 2
        # q_plus = Sign.apply(F.relu(q_plus))
        # q_minus_low = Sign.apply(F.relu(q_minus_low))
        # q_minus_high = Sign.apply(F.relu(q_minus_high))
        q_plus = F.relu(q_int)
        q_minus_low = F.relu(-q_int)
        q_minus_high = F.relu(-q_int-1)
        q_minus_low = q_minus_low - 2 * q_minus_high
        q_plus = q_plus.view((q_w.size(0), q_w.size(1) * q_w.size(2)))
        q_minus_high = q_minus_high.view((q_w.size(0), q_w.size(1) * q_w.size(2)))
        q_minus_low = q_minus_low.view((q_w.size(0), q_w.size(1) * q_w.size(2)))
        q_quant = q_int.view((q_w.size(0), q_w.size(1) * q_w.size(2)))
        unfold = nn.Unfold(kernel_size=(1, 7))
        x = F.conv1d(x, q_w,
                     stride=model.qconv1.conv_module.stride,
                     padding=model.qconv1.conv_module.padding, dilation=model.qconv1.conv_module.dilation,
                     groups=model.qconv1.conv_module.groups)
        bias = F.conv1d(Input_Bias, q_int,
                     stride=model.qconv1.conv_module.stride,
                     padding=model.qconv1.conv_module.padding, dilation=model.qconv1.conv_module.dilation,
                     groups=model.qconv1.conv_module.groups)
        # x_int = x_int.view(x_int.size(0), x_int.size(1), 1, x_int.size(2))
        x_linear = unfold(x_int)
        x_linear = x_linear.view(x_int.size(0), 7, x_int.size(1), -1)
        x_linear = x_linear.transpose(1, 3)
        x_linear = torch.reshape(x_linear, (x_linear.size(0) * x_linear.size(1), x_linear.size(2) * x_linear.size(3)))
        GroupNum = int(np.ceil(x_linear.size(1) / 4))
        x_quant = torch.zeros((x_linear.size(0), q_plus.size(0)))
        for i in range(GroupNum):
            y = F.linear(x_linear[:, (i * 4):(i + 1) * 4], q_plus[:, (i * 4):(i + 1) * 4])
            index = torch.randint(0, 1000, y.size())
            res = ProbSample[y.to(torch.int), index] + y / 15 - (y / 15).detach()
            # x_quant += res * 15
            x_quant += y
            y = F.linear(x_linear[:, (i * 4):min((i + 1) * 4, x_linear.size(1))], q_minus_low[:, (i * 4):min((i + 1) * 4, x_linear.size(1))])
            index = torch.randint(0, 1000, y.size())
            res = ProbSample[y.to(torch.int), index] + y / 15 - (y / 15).detach()
            # x_quant += -res * 15
            x_quant += -y
            y = F.linear(x_linear[:, (i * 4):min((i + 1) * 4, x_linear.size(1))], q_minus_high[:, (i * 4):min((i + 1) * 4, x_linear.size(1))])
            index = torch.randint(0, 1000, y.size())
            res = ProbSample[y.to(torch.int), index] + y / 15 - (y / 15).detach()
            # x_quant += -2 * res * 15
            x_quant += -2 * y
        # x_plus = F.linear(x_linear, q_plus, bias=None)
        # x_minus_low = F.linear(x_linear, q_minus_low, bias=None)
        # x_minus_high = F.linear(x_linear, q_minus_high, bias=None)
        # x_quant = x_plus - x_minus_low - 2 * x_minus_high
        # x_quant = F.linear(x_linear, q_quant, bias=None)
        x_quant = x_quant.view(x.size(0), -1, x.size(1))
        x_quant = x_quant.transpose(1, 2)
        OutputData = x_quant.clone()
        x_quant = x_quant - bias
        x_quant = x_quant / 16
        x = model.qconv1.fake_o.apply(x_quant, model.qconv1.qo)
        x = model.qrelu1.quantize_inference(x)
        x = model.qmaxpool1d_1.quantize_inference(x)

        x = x.view(-1, 4)
        x = model.qfc.quantize_inference(x)
        # x = model.sigmoid(x)
        # output = model(data)
        output = x.cpu().detach().numpy()
        target = target.cpu().detach().numpy()
        TP_TEMP, FP_TEMP, TN_TEMP, FN_TEMP = post_process(output, target)
        if TP_TEMP != 0 or FN_TEMP != 0:
            log.append(count)
        TP_O += TP_TEMP
        FP_O += FP_TEMP
        TN_O += TN_TEMP
        FN_O += FN_TEMP
        """
        temp = output.reshape(-1)
        x_axis = np.arange(0, temp.shape[0])
        plt.scatter(x_axis, output)
        plt.title('test_data')
        plt.show()
        """
        # DeployPlus = q_plus[0:2].detach().numpy()
        # DeployMinusHigh = q_minus_high[0:2].detach().numpy()
        # DeployMinusLow = q_minus_low[0:2].detach().numpy()
        DeployPlus = q_plus.detach().numpy()
        DeployMinusHigh = q_minus_high.detach().numpy()
        DeployMinusLow = q_minus_low.detach().numpy()

        DeployPlusHigh = np.zeros(DeployPlus.shape)
        DeployList = [DeployPlusHigh, DeployMinusHigh, DeployPlus, DeployMinusLow]
        WeightList = []
        for i in range(DeployPlus.shape[0]):
            for item in DeployList:
                for j in range(DeployPlus.shape[1]):
                    WeightList.append(item[i, j])
                for _ in range(2):
                    WeightList.append(0)
        COMPUTE_GROUP = int(np.ceil(DeployPlus.shape[1] / Parallellism))
        InputData = x_linear.detach().numpy()
        InputData = np.hstack([InputData, np.zeros((InputData.shape[0], 2))])
        AvailablePlace = [2, 3, 2]
        StorePlace = [7, 1, 2, 1]
        encoder = Encoder(Input_Dim=x_dim, Hidden_Dim=hidden_dim, Hidden_Dim2=hidden_dim2,
                        Hidden_Dim3=hidden_dim3, Latent_Dim=latent_dim)
        decoder = Decoder(Latent_Dim=latent_dim, Hidden_Dim=hidden_dim, Hidden_Dim2=hidden_dim2,
                        Hidden_Dim3=hidden_dim3, Output_Dim=x_dim)
        model_empty = Model(ENCODER=encoder, DECODER=decoder).to(DEVICE)
        model_empty.Decoder.FC_hidden.InMin = torch.nn.Parameter(torch.tensor(0.), requires_grad=False)
        model_empty.Decoder.FC_hidden.InMax = torch.nn.Parameter(torch.tensor(15.), requires_grad=False)
        model_empty.eval()
        Hardware_Result = Parallel_Compute_Serial_Ctrl(InputData, AvailablePlace, WeightList, model_empty, StorePlace, Result_Dim=2 * 2, GROUP_TOTAL=COMPUTE_GROUP)
        True_Result = np.zeros((OutputData.size(0), 2, OutputData.size(2)))
        for batch in range(True_Result.shape[0]):
            for channel in range(True_Result.shape[1]):
                for index in range(True_Result.shape[2]):
                    HardwareX = batch * True_Result.shape[2] + index
                    True_Result[batch, channel, index] += Hardware_Result[HardwareX, channel * 2] * 2 + Hardware_Result[HardwareX, channel * 2 + 1]
        True_Result = True_Result * 15
        OutputData = OutputData.detach().numpy()
        # print((OutputData[:, 0:2, :] - True_Result).min(), (OutputData[:, 0:2, :] - True_Result).max())
        # OutputData[:, 0:2, :] = True_Result
        print((OutputData - True_Result).min(), (OutputData - True_Result).max())
        OutputData = True_Result
        OutputData = torch.tensor(OutputData, device=device)
        x = OutputData - bias
        x = x / 16
        x = model.qconv1.fake_o.apply(x, model.qconv1.qo)
        x = model.qrelu1(x)
        x = model.qmaxpool1d_1(x)

        x = x.view(-1, 4)
        x = model.qfc.quantize_inference(x)
        # x = model.sigmoid(x)
        # output = model(data)
        output = x.cpu().detach().numpy()
        # target = target.cpu().detach().numpy()
        TP_TEMP, FP_TEMP, TN_TEMP, FN_TEMP = post_process(output, target)
        TP += TP_TEMP
        FP += FP_TEMP
        TN += TN_TEMP
        FN += FN_TEMP
        print("Hardware Deployed")
        print("TP, FP, TN, FN", TP, FP, TN, FN)
        print("accuracy", (TP+TN)/(TP+FP+TN+FN))
        # print("sensitivity", TP / (TP + FN))
        
        print("Software")
        print("TP, FP, TN, FN", TP_O, FP_O, TN_O, FN_O)
        print("accuracy", (TP_O+TN_O)/(TP_O+FP_O+TN_O+FN_O))
        # print("sensitivity", TP_O / (TP_O + FN_O))
        break
    # print("Target:", log)
    # print(check[0])
    # print(q_data.size(), q_w.size(), OutputData.size())
    return


def train_VAE():
    encoder = Encoder(Input_Dim=x_dim, Hidden_Dim=hidden_dim, Hidden_Dim2=hidden_dim2,
                      Hidden_Dim3=hidden_dim3, Latent_Dim=latent_dim)
    decoder = Decoder(Latent_Dim=latent_dim, Hidden_Dim=hidden_dim, Hidden_Dim2=hidden_dim2,
                      Hidden_Dim3=hidden_dim3, Output_Dim=x_dim)
    model_old = torch.load("C:/Users/16432/Desktop/Workplace/Python/VAE_DEMO/checkpoint_gamma_ternary_91.ptr", map_location=DEVICE)
    model = Model(ENCODER=encoder, DECODER=decoder).to(DEVICE)
    RefDict = model_old.state_dict()
    CorrectDict = dict()
    for that in RefDict:
        print(that)
        if that == "Decoder.FC_hidden.weight":
            CorrectDict["Decoder.FC_hidden.layer.weight"] = RefDict[that]
        else:
            CorrectDict[that] = RefDict[that]
    # CorrectDict["Decoder.FC_hidden.max"] = model.state_dict()["Decoder.FC_hidden.max"]
    # CorrectDict["Decoder.FC_hidden.min"] = model.state_dict()["Decoder.FC_hidden.min"]
    # CorrectDict["Decoder.FC_hidden.InMax"] = model.state_dict()["Decoder.FC_hidden.InMax"]
    # CorrectDict["Decoder.FC_hidden.InMin"] = model.state_dict()["Decoder.FC_hidden.InMin"]
    model.load_state_dict(CorrectDict)
    optimizer = Adam(model.parameters(), lr=lr)

    print("Start training VAE...")
    model.train()
    BestScore = 10000
    for epoch in range(epochs):
        overall_loss = 0
        for batch_idx, (x, _) in enumerate(train_loader):
            x = x.view(batch_size, x_dim)
            x = x.to(DEVICE)
            x = torch.round((x - x.min()) / (x.max() - x.min()) * 2) / 2
            optimizer.zero_grad()

            x_hat, mean, log_var = model(x)
            loss = loss_function(x, x_hat, mean, log_var)
            # print(loss, overall_loss)
            overall_loss += loss.item()

            loss.backward()
            optimizer.step()

        # if epoch == 10:
        #     for item in optimizer.param_groups:
        #         item["lr"] *= 0.1
        Score = overall_loss / (batch_idx * batch_size)
        print("\tEpoch", epoch + 1, "complete!", "\tAverage Loss: ", Score)
        if Score < BestScore:
            torch.save(model, './checkpoint_test_' + str(epoch) + '.ptr')
            with torch.no_grad():
                noise = torch.randn(batch_size, latent_dim).to(DEVICE)
                generated_images = model.Decoder(noise)
                save_image(generated_images.view(batch_size, 1, 28, 28), 'generated_sample.png')
            BestScore = Score

    print("Finish!!")

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
    return


def main():
    # CNN_DEMO()
    Communication_test()
    # VAE_DEMO()
    # Write_Test(ENABLE_BYTE=0x01, ADDR_BYTE=0x07, FileNameA="ResC", FileNameB="ResD")
    # WL_LIST_SUB, BL_LIST_SUB, _ = FindMatchGroup(FileNameA="ResC", FileNameB="ResD", GROUP=1)
    return


if __name__ == '__main__':
    main()
