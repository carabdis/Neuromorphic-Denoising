
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

dataset_path = './Analog_Memory_DEMO/VAE_DEMO/datasets'

# Analog buffer distribution for neural network training
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
        self.InMax = nn.Parameter(torch.tensor(0., device=DEVICE))
        self.InMin = nn.Parameter(torch.tensor(10., device=DEVICE))
        self.m = torch.nn.ReLU()

    def forward(self, Input):
        w = Sign.apply(self.layer.weight)
        wPlus = self.m(w)
        wMinus = self.m(-w)
        if self.training:
            self.InMax = nn.Parameter(torch.max(Input).detach())
            self.InMin = nn.Parameter(torch.min(Input).detach())
        else:
            Input = torch.clip(Round.apply((Input - self.InMin) / (self.InMax - self.InMin) * 15), 0, 15)
        Min = torch.ones(Input.size(), device=DEVICE) * self.InMin
        bias = torch.nn.functional.linear(Min, w)
        OutList = []
        x = torch.zeros((Input.size()[0], hidden_dim3), device=DEVICE)
        for i in range(4):
            y = torch.nn.functional.linear(Input[:, (i * 4):(i + 1) * 4], wPlus[:, (i * 4):(i + 1) * 4])
            OutList.append(y)
            y = torch.nn.functional.linear(Input[:, (i * 4):(i + 1) * 4], wMinus[:, (i * 4):(i + 1) * 4])
            OutList.append(-y)
        for i in range(len(OutList)):
            tensor = OutList[i]
            x = (x + torch.clip(Round.apply(tensor / 15), -4, 4)
                 * (self.InMax - self.InMin))
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
    while result <= 0:
        result = aa_i2c_write(handle, DEVICE, mode, data)
        TryCount += 1
    return TryCount >= Limit

def Communication_Read(handle, data, DEVICE, mode, Limit = 10):
    TryCount = 0
    result = 0
    while result <= 0:
        result, read_byte = aa_i2c_read(handle, DEVICE, mode, data)
        TryCount += 1
        # aa_sleep_ms(1)
    return TryCount >= Limit

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
        # if RawResult > 200000:
        #     NotSuccess = False
        #     return None
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
    Result_Batch = np.zeros((Data.shape[0], Result_Dim))
    Test_Result = 0
    Sign = 1
    ratio = 0
    monument = 0.1
    for i in range(int(len(Weight) / Parallelism)):
        data_LIST = []
        data_TEMP = 0
        # Write In Weight
        ratio = i / (int(len(Weight) / Parallelism))
        if ratio > monument:
            monument = min(ratio + 0.1, 1)
            print(i, '/', int(len(Weight) / Parallelism))
        if i % 300 == 0:
            time.sleep(1)
        for j in range(Parallelism):
            if Weight[Parallelism * i + j] == Initial_Weight[j]:
                continue
            else:
                Enable = int(2 ** j)
                Enable <<= GROUP
                Enable += int(2 ** COL)
                NotSuccess= Write(handle, Weight[Parallelism * i + j], Symbol, Enable,
                                          DEVICE_COMM, weight_addr, MODE_BYTE, CHECK_BYTE, ADDR_BYTE, ENABLE_BYTE)
                Write_Message = array('B', [CHECK_BYTE])
                Symbol = (Symbol + 1) % 2
                HandShake(handle, DEVICE_COMM, Symbol)
                Initial_Weight[j] = Weight[Parallelism * i + j]
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
            # Read Out Result
            NotSuccess = Communication_Write(handle=handle, data=result_addr, DEVICE=DEVICE_COMM, mode=AA_I2C_NO_STOP)
            NotSuccess = Communication_Read(handle=handle, data=readin, DEVICE=DEVICE_COMM, mode=AA_I2C_NO_FLAGS)
            Result_Batch[j, int(i / GROUP_TOTAL / 2)] += Sign * min(3, readin[0])
            Test_Result = 0
        if i % GROUP_TOTAL == GROUP_TOTAL - 1:
            Sign = -Sign
    aa_close(handle)
    return Result_Batch

def DeployWeightLog(DeployWeight):
    with open("DeployWeight.csv", "w", encoding='utf-8', newline='') as f:
        csv_writer = csv.writer(f)
        for item in DeployWeight:
            csv_writer.writerow(item)
        f.close()
    return

def VAE_DEMO():
    # MNIST Generative Model
    model = torch.load('./Analog_Memory_DEMO/VAE_DEMO/checkpoint_test_with_hardware.ptr', map_location=DEVICE)
    weight = Sign.apply(model.Decoder.FC_hidden.layer.weight).detach().numpy().transpose()
    DeployList = np.zeros((weight.shape[0], 2 * weight.shape[1]))
    weight = weight.transpose()
    for i in range(weight.shape[1]):
        for j in range(weight.shape[0]):
            if weight[j, i] == 1:
                DeployList[i][2 * j] = 1
            elif weight[j, i] == -1:
                DeployList[i][2 * j + 1] = 1
    print(DeployList.shape)
    DeployWeightLog(DeployList)
    WeightList = []
    DataChannel = DeployList.shape[0]
    DeployChannel = DeployList.shape[1]
    for i in range(DeployChannel):
        for j in range(DataChannel):
            WeightList.append(DeployList[j][i])
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
            print(batch_idx, label)
            x = x.view(batch_size, x_dim)
            x = x.to(DEVICE)
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
    AvailablePlace = [2, 3, 2]
    StorePlace = [7, 3, 2, 1]
    with torch.no_grad():
        overall_loss = 0
        HardwareResult = Parallel_Compute_Serial_Ctrl(BatchData, AvailablePlace, WeightList, model, StorePlace)
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
    x_hat = torch.cat([x_hat_active.view(28, 28), x_hat.view(28, 28)], 1)
    print(x_hat.size())
    show_image(x_hat, idx=0)
    point = overall_loss_origin.item() / count / batch_size
    point = '%.4f' % point
    point = overall_loss / count / batch_size
    point = '%.4f' % point
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
    batch_size = 32
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    
    model = Net().to(device)
    model.quantize(1, 2, 2, 4)
    model.load_state_dict(torch.load(('./Analog_Memory_DEMO/cnn_fixpoint/ckpt/1d_cnn_2bit_sdj_avgpool.pt'), map_location='cpu'))
    count = 0
    pos = 0
    neg = 0
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
            count += batch_size
            continue
        print("Begin Number", count, pos, neg)
        data, target = data.to(device), target.to(device)
        x = model.qconv1.fake_i.apply(data, model.qconv1.qi)
        x_int = x * 8 + 6
        Input_Bias = 6 * torch.ones(x.size(), device=DEVICE)
        q_w = FakeQuantize.apply(model.qconv1.conv_module.weight, model.qconv1.qw)
        q_int = q_w * 2
        q_plus = F.relu(q_int)
        q_minus_low = F.relu(-q_int)
        q_minus_high = F.relu(-q_int-1)
        q_minus_low = q_minus_low - 2 * q_minus_high
        q_plus = q_plus.view((q_w.size(0), q_w.size(1) * q_w.size(2)))
        q_minus_high = q_minus_high.view((q_w.size(0), q_w.size(1) * q_w.size(2)))
        q_minus_low = q_minus_low.view((q_w.size(0), q_w.size(1) * q_w.size(2)))
        unfold = nn.Unfold(kernel_size=(1, 7))
        x = F.conv1d(x, q_w,
                     stride=model.qconv1.conv_module.stride,
                     padding=model.qconv1.conv_module.padding, dilation=model.qconv1.conv_module.dilation,
                     groups=model.qconv1.conv_module.groups)
        bias = F.conv1d(Input_Bias, q_int,
                     stride=model.qconv1.conv_module.stride,
                     padding=model.qconv1.conv_module.padding, dilation=model.qconv1.conv_module.dilation,
                     groups=model.qconv1.conv_module.groups)
        x_linear = unfold(x_int)
        x_linear = x_linear.view(x_int.size(0), 7, x_int.size(1), -1)
        x_linear = x_linear.transpose(1, 3)
        x_linear = torch.reshape(x_linear, (x_linear.size(0) * x_linear.size(1), x_linear.size(2) * x_linear.size(3)))
        GroupNum = int(np.ceil(x_linear.size(1) / 4))
        x_quant = torch.zeros((x_linear.size(0), q_plus.size(0)))
        for i in range(GroupNum):
            y = F.linear(x_linear[:, (i * 4):(i + 1) * 4], q_plus[:, (i * 4):(i + 1) * 4])
            index = torch.randint(0, 1000, y.size())
            x_quant += y
            y = F.linear(x_linear[:, (i * 4):min((i + 1) * 4, x_linear.size(1))], q_minus_low[:, (i * 4):min((i + 1) * 4, x_linear.size(1))])
            index = torch.randint(0, 1000, y.size())
            x_quant += -y
            y = F.linear(x_linear[:, (i * 4):min((i + 1) * 4, x_linear.size(1))], q_minus_high[:, (i * 4):min((i + 1) * 4, x_linear.size(1))])
            index = torch.randint(0, 1000, y.size())
            x_quant += -2 * y
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
        output = x.cpu().detach().numpy()
        target = target.cpu().detach().numpy()
        TP_TEMP, FP_TEMP, TN_TEMP, FN_TEMP = post_process(output, target)
        if TP_TEMP != 0 or FN_TEMP != 0:
            log.append(count)
        TP_O += TP_TEMP
        FP_O += FP_TEMP
        TN_O += TN_TEMP
        FN_O += FN_TEMP
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
        output = x.cpu().detach().numpy()
        TP_TEMP, FP_TEMP, TN_TEMP, FN_TEMP = post_process(output, target)
        TP += TP_TEMP
        FP += FP_TEMP
        TN += TN_TEMP
        FN += FN_TEMP
        print("Hardware Deployed")
        print("TP, FP, TN, FN", TP, FP, TN, FN)
        print("accuracy", (TP+TN)/(TP+FP+TN+FN))
        
        print("Software")
        print("TP, FP, TN, FN", TP_O, FP_O, TN_O, FN_O)
        print("accuracy", (TP_O+TN_O)/(TP_O+FP_O+TN_O+FN_O))
        break
    return


def train_VAE():
    encoder = Encoder(Input_Dim=x_dim, Hidden_Dim=hidden_dim, Hidden_Dim2=hidden_dim2,
                      Hidden_Dim3=hidden_dim3, Latent_Dim=latent_dim)
    decoder = Decoder(Latent_Dim=latent_dim, Hidden_Dim=hidden_dim, Hidden_Dim2=hidden_dim2,
                      Hidden_Dim3=hidden_dim3, Output_Dim=x_dim)
    model_old = torch.load("./Analog_Memory_DEMO/VAE_DEMO/checkpoint_gamma_ternary_91.ptr", map_location=DEVICE)
    model = Model(ENCODER=encoder, DECODER=decoder).to(DEVICE)
    RefDict = model_old.state_dict()
    CorrectDict = dict()
    for that in RefDict:
        print(that)
        if that == "Decoder.FC_hidden.weight":
            CorrectDict["Decoder.FC_hidden.layer.weight"] = RefDict[that]
        else:
            CorrectDict[that] = RefDict[that]
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
            overall_loss += loss.item()

            loss.backward()
            optimizer.step()

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
    Communication_test()    # I2C Communication process test
    # CNN_DEMO()            # 1D-CNN Demo for Epilpsey Detection
    # VAE_DEMO()            # MNIST generative model
    return


if __name__ == '__main__':
    main()
