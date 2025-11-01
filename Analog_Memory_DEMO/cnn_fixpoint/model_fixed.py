"""
Author: CSuperlei
Date: 2022-10-08 11:11:27
LastEditTime: 2022-10-10 22:22:50
Description:
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import matplotlib.pyplot as plt
import numpy as np


class FakeQuantize(Function):  ## 伪量化(量化感知)
    @staticmethod
    def forward(ctx, x, qparam):
        x = qparam.quantize_tensor(x)
        x = qparam.dequantize_tensor(x)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class Float2Fixed(nn.Module):  ## 浮点数转定点数二进制表示 +0.00
    def __init__(self, Qn=2, num_bits=4, plus=False) -> None:
        super().__init__()
        self.num_bits = num_bits - 1  ## 除去符号位剩余位数(包括 整数 + 小数)
        self.Qn = Qn  ## 小数部分
        self.plus = plus

    def forward(self, input):
        if self.plus:
            upb = pow(2, self.num_bits - self.Qn + 1) - 1 / pow(2, self.Qn)
            downb = 0
        else:
            upb = pow(2, self.num_bits - self.Qn) - 1 / pow(2, self.Qn)
            downb = -1 * pow(2, self.num_bits - self.Qn)
        x = torch.clamp(input, downb, upb)
        x = x * pow(2, self.Qn)
        out = torch.round(x)
        return out


class Fixed2Float(nn.Module):  ## 定点数二进制转定点数小数
    def __init__(self, Qn=2, num_bits=4) -> None:
        super().__init__()
        self.Qn = Qn
        self.num_bits = num_bits

    def forward(self, input):
        out = input / pow(2, self.Qn)
        return out

    
def gaussian_noise(inputs, mean, std):  ## generate gaussian noise
    noise = torch.randn(inputs.size())*std + mean
    return noise


def invert_gaussian(inputs, mean, std):  ## generate gaussian noise
    invert = 1/inputs
    invert_gaussian = invert + gaussian_noise(inputs, mean, std)
    out = 1/invert_gaussian
    return out


def quantize_tensor(x, Qn, num_bits, plus):  ##量化
    convert = Float2Fixed(Qn, num_bits, plus)
    q_x = convert(x)
    return q_x


def dequantize_tensor(q_x, Qn, num_bits):  ## 反量化
    convert = Fixed2Float(Qn, num_bits)
    x = convert(q_x)
    return x


class QParam(nn.Module):
    def __init__(self, Qn, num_bits, plus=False) -> None:
        super().__init__()
        self.Qn = Qn
        self.num_bits = num_bits
        self.plus = plus

    def quantize_tensor(self, tensor):
        return quantize_tensor(tensor, self.Qn, self.num_bits, self.plus)

    def dequantize_tensor(self, q_x):
        return dequantize_tensor(q_x, self.Qn, self.num_bits)


class QConv1d(nn.Module):

    def __init__(self, conv_module, Qn, num_bits, Qn_io, num_bits_io, plus=False) -> None:
        super().__init__()
        self.conv_module = conv_module
        self.num_bits = num_bits
        self.Qn = Qn
        self.Qn_io = Qn_io
        self.num_bits_io = num_bits_io
        self.plus = plus
        self.qw = QParam(self.Qn, self.num_bits, self.plus)  ## 卷积层权重量化
        self.qi = QParam(self.Qn_io, self.num_bits_io, self.plus)  ## 卷积层输入量化
        self.qo = QParam(2, 6, self.plus)  ## 卷积层输出量化
        self.fake_i = FakeQuantize()
        self.fake_o = FakeQuantize()

    def forward(self, x):
        x = self.fake_i.apply(x, self.qi)
        q_w = FakeQuantize.apply(self.conv_module.weight, self.qw)
        x = F.conv1d(x, q_w,
                     stride=self.conv_module.stride,
                     padding=self.conv_module.padding, dilation=self.conv_module.dilation,
                     groups=self.conv_module.groups)

        x = self.fake_o.apply(x, self.qo)
        return x
    
    def QGinfer(self, x, mean, std, invert=True):
        x = self.fake_i.apply(x, self.qi)
        q_w = FakeQuantize.apply(self.conv_module.weight, self.qw)
        if invert:
            q_g_w = invert_gaussian(q_w, mean, std)
        else:
            q_g_w = q_w + gaussian_noise(q_w, mean, std)
        x = F.conv1d(x, q_g_w,
                     stride=self.conv_module.stride,
                     padding=self.conv_module.padding, dilation=self.conv_module.dilation,
                     groups=self.conv_module.groups)

        x = self.fake_o.apply(x, self.qo)
        return x

    def freeze(self):
        self.conv_module.weight.data = self.qw.quantize_tensor(self.conv_module.weight.data)
        self.conv_module.weight.data = self.qw.dequantize_tensor(self.conv_module.weight.data)

    def quantize_gaussian_inference(self, x, mean, std, invert=True):
        x = self.fake_i.apply(x, self.qi)
        # temp = x.cpu().detach().numpy()
        # np.save('D:/PycharmProjects/1d_cnn_fixpoint/1D_CNN_dataset/chb01/test0/chb01_03_0_s_fake_int.npy',temp)

        ## use QG infer
        x = self.QGinfer(x, mean, std, invert)

        return x
    
    def quantize_inference(self, x):
        x = self.fake_i.apply(x, self.qi)
        # temp = x.cpu().detach().numpy()
        # np.save('D:/PycharmProjects/1d_cnn_fixpoint/1D_CNN_dataset/chb01/test0/chb01_03_0_s_fake_int.npy',temp)
        ##
        '''
        temp = x.cpu().detach().numpy()
        temp = temp.reshape(-1)
        x_axis = np.arange(0, temp.shape[0])
        plt.scatter(x_axis, temp)
        plt.title('input_data')
        plt.show()
        plt.clf()
        '''
        ## original
        x = self.conv_module(x)
        x = self.fake_o.apply(x, self.qo)

        return x


class QLinear(nn.Module):
    def __init__(self, fc_module, Qn, num_bits, Qn_io, num_bits_io) -> None:
        super().__init__()
        self.fc_module = fc_module
        self.num_bits = num_bits
        self.Qn = Qn
        self.Qn_io = Qn_io
        self.num_bits_io = num_bits_io

    def forward(self, x):
        x = F.linear(x, self.fc_module.weight)
        return x

    def freeze(self):  ## fc层不量化
        pass

    def quantize_inference(self, x):
        x = self.fc_module(x)
        return x


class QReLU(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        pass

    def forward(self, x):
        x = F.relu(x)
        return x

    def quantize_inference(self, x):
        x = x.clone()
        x[x < 0] = 0
        return x


class QMaxPooling1d(nn.Module):
    def __init__(self, kernel_size=3, stride=1, padding=0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def freeze(self):
        pass

    def forward(self, x):
        x = F.max_pool1d(x, self.kernel_size, self.stride, self.padding)
        return x

    def quantize_inference(self, x):
        return F.max_pool1d(x, self.kernel_size, self.stride, self.padding)


class Net(nn.Module):
    def __init__(self, ):
        super(Net, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=18, out_channels=4, kernel_size=7, bias=False)
        # self.conv1 = nn.Conv1d(in_channels=18, out_channels=4, kernel_size=4, bias=False)
        self.linear = nn.Linear(in_features=4, out_features=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # print(x.size())
        x = self.conv1(x)
        # print(x.size())
        x = F.relu(x)
        # print(x.size())
        x = F.max_pool1d(x, 26)
        # print(x.size())
        x = x.view(-1, 4)
        # print(x.size())
        x = self.linear(x)
        # print(x.size())
        return x

    def quantize(self, Qn, num_bits, Qn_io, num_bits_io):
        self.qconv1 = QConv1d(self.conv1, Qn, num_bits, Qn_io, num_bits_io, plus=True)
        self.qrelu1 = QReLU()
        self.qmaxpool1d_1 = QMaxPooling1d(kernel_size=26, stride=2, padding=0)
        self.qfc = QLinear(self.linear, Qn, num_bits, Qn_io, num_bits_io)

    def quantize_forward(self, x):
        x = self.qconv1(x)
        x = self.qrelu1(x)
        x = self.qmaxpool1d_1(x)

        x = x.view(-1, 4)
        x = self.qfc(x)
        x = self.sigmoid(x)

        return x

    def freeze(self):
        self.qconv1.freeze()
        # self.qfc.freeze()

    def quantize_inference(self, x):
        '''
        量化输入
        '''

        x = self.qconv1.quantize_inference(x)

        ##
        '''
        temp = x.cpu().detach().numpy()
        temp = temp.reshape(-1)
        x_axis = np.arange(0, temp.shape[0])
        plt.scatter(x_axis, temp)
        plt.title('conv1_out')
        plt.show()
        plt.clf()
        '''
        ##
        x = self.qrelu1.quantize_inference(x)
        ##
        '''
        temp = x.cpu().detach().numpy()
        temp = temp.reshape(-1)
        x_axis = np.arange(0, temp.shape[0])
        plt.scatter(x_axis, temp)
        plt.title('relu_out')
        plt.show()
        plt.clf()
        '''
        ##

        x = self.qmaxpool1d_1.quantize_inference(x)
        ##
        '''
        temp = x.cpu().detach().numpy()
        temp = temp.reshape(-1)
        x_axis = np.arange(0, temp.shape[0])
        plt.scatter(x_axis, temp)
        plt.title('max_out')
        plt.show()
        plt.clf()
        '''
        ##
        x = x.view(-1, 4)
        x = self.qfc.quantize_inference(x)
        ##

        # temp = x.cpu().detach().numpy()
        # temp = temp.reshape(-1)
        # x_axis = np.arange(0, temp.shape[0])
        # plt.scatter(x_axis, temp)
        # plt.title('qfc_out')
        # plt.show()
        # plt.clf()

        ##
        # print(x.shape)

        # x = self.sigmoid(x)

        return x
    
    def quantize_gaussian_inference(self, x, mean, std, invert=True):
        '''
        量化输入
        '''

        x = self.qconv1.quantize_gaussian_inference(x, mean, std, invert)

        ##
        '''
        temp = x.cpu().detach().numpy()
        temp = temp.reshape(-1)
        x_axis = np.arange(0, temp.shape[0])
        plt.scatter(x_axis, temp)
        plt.title('conv1_out')
        plt.show()
        plt.clf()
        '''
        ##
        x = self.qrelu1.quantize_inference(x)
        ##
        '''
        temp = x.cpu().detach().numpy()
        temp = temp.reshape(-1)
        x_axis = np.arange(0, temp.shape[0])
        plt.scatter(x_axis, temp)
        plt.title('relu_out')
        plt.show()
        plt.clf()
        '''
        ##

        x = self.qmaxpool1d_1.quantize_inference(x)
        ##
        '''
        temp = x.cpu().detach().numpy()
        temp = temp.reshape(-1)
        x_axis = np.arange(0, temp.shape[0])
        plt.scatter(x_axis, temp)
        plt.title('max_out')
        plt.show()
        plt.clf()
        '''
        ##
        x = x.view(-1, 4)
        x = self.qfc.quantize_inference(x)
        ##

        # temp = x.cpu().detach().numpy()
        # temp = temp.reshape(-1)
        # x_axis = np.arange(0, temp.shape[0])
        # plt.scatter(x_axis, temp)
        # plt.title('qfc_out')
        # plt.show()
        # plt.clf()

        ##
        # print(x.shape)

        # x = self.sigmoid(x)

        return x
