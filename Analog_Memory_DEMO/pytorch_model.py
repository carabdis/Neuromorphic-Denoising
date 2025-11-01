import torch
import torch.nn as nn
from torch.autograd import Function
import numpy as np


cuda = torch.cuda.is_available()
DEVICE = torch.device("cuda" if cuda else "cpu")


np.set_printoptions(threshold=np.inf, linewidth=np.inf)
Prob = [[1.0, 0.0, 0.0, 0.0],
		[1.0, 0.0, 0.0, 0.0],
		[1.0, 0.0, 0.0, 0.0],
		[1.0, 0.0, 0.0, 0.0],
		[1.0, 0.0, 0.0, 0.0],
		[1.0, 0.0, 0.0, 0.0],
		[1.0, 0.0, 0.0, 0.0],
		[1.0, 0.0, 0.0, 0.0],
		[1.0, 0.0, 0.0, 0.0],
		[1.0, 0.0, 0.0, 0.0],
		[1.0, 0.0, 0.0, 0.0],
		[1.0, 0.0, 0.0, 0.0],
		[1.0, 0.0, 0.0, 0.0],
		[1.0, 0.0, 0.0, 0.0],
		[1.0, 0.0, 0.0, 0.0],
		[1.0, 0.0, 0.0, 0.0],
		[0.06, 0.91, 0.03, 0.0],
		[0.0, 0.9, 0.1, 0.0],
		[0.02, 0.89, 0.09, 0.0],
		[0.0, 0.91, 0.09, 0.0],
		[0.04, 0.93, 0.03, 0.0],
		[0.03, 0.92, 0.05, 0.0],
		[0.05, 0.81, 0.13, 0.01],
		[0.04, 0.9, 0.06, 0.0],
		[0.02, 0.91, 0.07, 0.0],
		[0.02, 0.97, 0.01, 0.0],
		[0.0, 0.96, 0.04, 0.0],
		[0.0, 0.74, 0.25, 0.01],
		[0.0, 0.81, 0.19, 0.0],
		[0.01, 0.89, 0.1, 0.0],
		[0.01, 0.91, 0.08, 0.0],
		[0.0, 0.77, 0.23, 0.0],
		[0.0, 0.42, 0.53, 0.05],
		[0.0, 0.53, 0.44, 0.03],
		[0.0, 0.74, 0.25, 0.01],
		[0.0, 0.46, 0.5, 0.04],
		[0.0, 0.43, 0.49, 0.08],
		[0.0, 0.53, 0.47, 0.0],
		[0.0, 0.6, 0.39, 0.01],
		[0.0, 0.46, 0.52, 0.02],
		[0.0, 0.71, 0.28, 0.01],
		[0.01, 0.67, 0.32, 0.0],
		[0.0, 0.49, 0.49, 0.02],
		[0.0, 0.44, 0.54, 0.02],
		[0.0, 0.4, 0.56, 0.04],
		[0.0, 0.23, 0.7, 0.07],
		[0.0, 0.1, 0.57, 0.33],
		[0.0, 0.2, 0.53, 0.27],
		[0.0, 0.14, 0.38, 0.48],
		[0.0, 0.08, 0.56, 0.36],
		[0.0, 0.26, 0.61, 0.13],
		[0.0, 0.22, 0.44, 0.34],
		[0.0, 0.08, 0.43, 0.49],
		[0.0, 0.02, 0.26, 0.72],
		[0.0, 0.01, 0.12, 0.87],
		[0.0, 0.01, 0.19, 0.8],
		[0.0, 0.0, 0.02, 0.98],
		[0.0, 0.01, 0.04, 0.95],
		[0.0, 0.0, 0.01, 0.99],
		[0.0, 0.0, 0.0, 1.0],
		[0.0, 0.0, 0.0, 1.0]]
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


class QuantConv(nn.Module):

    def __init__(self, Input_Dim, Output_Dim, Kernel, Stride, Bias):
        super(QuantConv, self).__init__()
        self.layer = nn.Conv2d(Input_Dim, Output_Dim, kernel_size=Kernel, stride=Stride, bias=Bias)
        self.InMax = nn.Parameter(torch.tensor(0., device=DEVICE))
        self.InMin = nn.Parameter(torch.tensor(10., device=DEVICE))
        self.m = torch.nn.ReLU()

    def forward(self, Input):
        Input = nn.functional.unfold(Input, self.layer.kernel_size, self.layer.dilation, self.layer.padding, 
                                     self.layer.stride).transpose(1, 2)
        BatchSize = Input.size(0)
        Input = Input.reshape((-1, Input.size(2)))
        w = torch.reshape(self.layer.weight, (self.layer.weight.size(0), -1))
        w = Sign.apply(w)
        wPlus = self.m(w)
        wMinus = self.m(-w)
        Input = torch.clip(Round.apply((Input - self.InMin) / (self.InMax - self.InMin) * 15), 0, 15)
        Min = torch.ones(Input.size(), device=DEVICE) * self.InMin
        bias = torch.nn.functional.linear(Min, w)
        OutList = []
        x = torch.zeros((Input.size()[0], w.size()[0]), device=DEVICE)
        for i in range(int(np.ceil(Input.size()[1] / 4))):
            y = torch.nn.functional.linear(Input[:, (i * 4):(i + 1) * 4], wPlus[:, (i * 4):(i + 1) * 4])
            OutList.append(y)
            y = torch.nn.functional.linear(Input[:, (i * 4):(i + 1) * 4], wMinus[:, (i * 4):(i + 1) * 4])
            OutList.append(-y)
        for i in range(len(OutList)):
            tensor = OutList[i]  # + (OutList[i] * torch.rand(OutList[i].size(), device=DEVICE) * 0.2).detach()
            LongTensor = torch.clip(Round.apply(tensor), -60, 60).long()
            QuantTensor = torch.clip(Round.apply(tensor / 15), -3, 3)
            RandTensor = torch.randint(0, 1000, QuantTensor.size(), device=DEVICE)
            TempTensor = torch.reshape(ProbSample[torch.abs(LongTensor.flatten()), RandTensor.flatten()],
                                       QuantTensor.size())
            AddTensor = TempTensor * torch.sign(QuantTensor) + QuantTensor - QuantTensor.detach()
            # AddTensor = QuantTensor
            x = (x + AddTensor  # * self.scale / 15
                 * (self.InMax - self.InMin))
        x = x + bias + self.layer.bias
        x = x.reshape((BatchSize, -1, x.size(1)))
        dim_value = int(np.sqrt(x.size(1)))
        x = x.transpose(1, 2).reshape(x.size(0), x.size(2), dim_value, dim_value)
        return x


class Encoder(nn.Module):
    def __init__(self, input_shape, latent_dim=100):
        super(Encoder, self).__init__()
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.Drop = nn.Dropout(p=0.2)
        
        # self.conv1 = nn.Conv2d(input_shape, 32, kernel_size=(4, 4), stride=2, bias=True)
        self.conv1 = QuantConv(input_shape, 32, Kernel=(4, 4), Stride=2, Bias=True)
        self.bn1 = nn.BatchNorm2d(32, momentum=0.01)
        self.act1 = nn.LeakyReLU()
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(4, 4), stride=2, bias=True)
        self.bn2 = nn.BatchNorm2d(64, momentum=0.01)
        self.act2 = nn.LeakyReLU()

        self.conv3 = nn.Conv2d(64, 128, kernel_size=(4, 4), stride=2, bias=True)
        self.bn3 = nn.BatchNorm2d(128, momentum=0.01)
        self.act3 = nn.LeakyReLU()

        self.conv4 = nn.Conv2d(128, 256, kernel_size=(4, 4), stride=2, bias=True)
        self.bn4 = nn.BatchNorm2d(256, momentum=0.01)
        self.act4 = nn.LeakyReLU()

        self.pool = nn.AdaptiveAvgPool2d(1)
        
        self.mean_linear = nn.Linear(256, latent_dim, bias=True)
        self.var_linear = nn.Linear(256, latent_dim, bias=True)
    
    def forward(self, x):
        x = self.Drop(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.act3(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.act4(x)

        x = self.pool(x)
        x = torch.flatten(x, start_dim=1)

        x_mean = self.mean_linear(x)
        x_var = self.var_linear(x)
        return x_mean, x_var
    
    def sample(self, mean, var):
        epsilon = torch.randn(mean.shape, device=DEVICE)
        return mean + torch.exp(0.5 * var) * epsilon


class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(latent_dim, 4096)
        
        self.upsample1 = nn.UpsamplingNearest2d(scale_factor=2)
        self.conv1 = nn.Conv2d(256, 128, kernel_size=(3, 3), stride=1, padding="same", bias=True)
        self.bn1 = nn.BatchNorm2d(128, momentum=0.01)
        self.act1 = nn.LeakyReLU()
        
        self.upsample2 = nn.UpsamplingNearest2d(scale_factor=2)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=(3, 3), stride=1, padding="same", bias=True)
        self.bn2 = nn.BatchNorm2d(64, momentum=0.01)
        self.act2 = nn.LeakyReLU()

        self.upsample3 = nn.UpsamplingNearest2d(scale_factor=2)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=(3, 3), stride=1, padding="same", bias=True)
        self.bn3 = nn.BatchNorm2d(32, momentum=0.01)
        self.act3 = nn.LeakyReLU()

        self.upsample4 = nn.UpsamplingNearest2d(scale_factor=2)
        self.conv4 = nn.Conv2d(32, 3, kernel_size=(3, 3), stride=1, padding="same", bias=True)
        self.act4 = nn.Sigmoid()
    
    def forward(self, x):
        x = self.linear1(x)
        x = torch.reshape(x, (-1, 256, 4, 4))
        x = self.upsample1(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.upsample2(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)

        x = self.upsample3(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.act3(x)

        x = self.upsample4(x)
        x = self.conv4(x)
        x = self.act4(x)
        return x


class VAE(nn.Module):
    def __init__(self, encoder, decoder, kl_weighting=1):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.kl_weighting = kl_weighting

    def forward(self, x):
        x_mean, x_var = self.encoder(x)
        latent = self.encoder.sample(x_mean, x_var)
        x_imp = self.decoder(latent)
        return x, x_imp, x_mean, x_var
