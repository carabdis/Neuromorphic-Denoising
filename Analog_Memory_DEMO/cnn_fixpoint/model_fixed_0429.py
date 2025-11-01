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
ListCount = np.zeros(4)
class FakeQuantize(Function):  ## 伪量化(量化感知)
    @staticmethod
    def forward(ctx, x, qparam):
        x = qparam.quantize_tensor(x)
        x = qparam.dequantize_tensor(x)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None

class Sign(Function):  ## 伪量化(量化感知)
    @staticmethod
    def forward(ctx, x):
        return torch.sign(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

class Round(Function):
    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

class Float2Fixed(nn.Module):  ## 浮点数转定点数二进制表示 +0.00
    def __init__(self, Qn=2, num_bits=4) -> None:
        super().__init__()
        self.num_bits = num_bits - 1  ## 除去符号位剩余位数(包括 整数 + 小数)
        self.Qn = Qn  ## 小数部分

    def forward(self, input):
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


def quantize_tensor(x, Qn, num_bits):  ##量化
    convert = Float2Fixed(Qn, num_bits)
    q_x = convert(x)
    return q_x


def dequantize_tensor(q_x, Qn, num_bits):  ## 反量化
    convert = Fixed2Float(Qn, num_bits)
    x = convert(q_x)
    return x


class QParam(nn.Module):
    def __init__(self, Qn, num_bits) -> None:
        super().__init__()
        self.Qn = Qn
        self.num_bits = num_bits

    def quantize_tensor(self, tensor):
        return quantize_tensor(tensor, self.Qn, self.num_bits)

    def dequantize_tensor(self, q_x):
        return dequantize_tensor(q_x, self.Qn, self.num_bits)


class QConv1d(nn.Module):

    def __init__(self, conv_module, Qn, num_bits, Qn_io, num_bits_io) -> None:
        super().__init__()
        self.conv_module = conv_module
        self.num_bits = num_bits
        self.Qn = Qn
        self.Qn_io = Qn_io
        self.num_bits_io = num_bits_io
        self.qw = QParam(self.Qn, self.num_bits)  ## 卷积层权重量化
        self.qi = QParam(self.Qn_io, self.num_bits_io)  ## 卷积层输入量化
        self.qo = QParam(2, 6)  ## 卷积层输出量化
        self.fake_i = FakeQuantize()
        self.fake_o = FakeQuantize()
        self.q_w_plus = torch.nn.Parameter(torch.relu(conv_module.weight.view((conv_module.weight.size(0), conv_module.weight.size(1) * conv_module.weight.size(2)))))
        self.q_w_minus_high = torch.nn.Parameter(torch.relu(-(conv_module.weight.view((conv_module.weight.size(0), conv_module.weight.size(1) * conv_module.weight.size(2))) + 0.5)))
        self.q_w_minus_low = torch.nn.Parameter(torch.relu(-conv_module.weight.view((conv_module.weight.size(0), conv_module.weight.size(1) * conv_module.weight.size(2))) + self.q_w_minus_high))

    def forward(self, x):
        x = self.fake_i.apply(x, self.qi)
        q_w = FakeQuantize.apply(self.conv_module.weight, self.qw)
        # x = F.conv1d(x, q_w,
        #              stride=self.conv_module.stride,
        #              padding=self.conv_module.padding, dilation=self.conv_module.dilation,
        #              groups=self.conv_module.groups)
        x_quant = self.hardware_action(x, q_w, self.q_w_plus, self.q_w_minus_low, self.q_w_minus_high)
        x = self.fake_o.apply(x_quant, self.qo)
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
    
    def hardware_action(self, x, q_w, q_plus, q_minus_low, q_minus_high):
        x_int = x * 8 + 6
        Input_Bias = 6 * torch.ones(x.size(), device=DEVICE)
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
                     stride=self.conv_module.stride,
                     padding=self.conv_module.padding, dilation=self.conv_module.dilation,
                     groups=self.conv_module.groups)
        bias = F.conv1d(Input_Bias, q_int,
                     stride=self.conv_module.stride,
                     padding=self.conv_module.padding, dilation=self.conv_module.dilation,
                     groups=self.conv_module.groups)
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
            x_quant += res * 15
            # x_quant += y
            # for item in y.flatten():
            #     ListCount[(torch.floor(item / 15)).flatten().to(torch.int)] += 1
            y = F.linear(x_linear[:, (i * 4):min((i + 1) * 4, x_linear.size(1))], q_minus_low[:, (i * 4):min((i + 1) * 4, x_linear.size(1))])
            index = torch.randint(0, 1000, y.size())
            res = ProbSample[y.to(torch.int), index] + y / 15 - (y / 15).detach()
            # for item in y.flatten():
            #     ListCount[(torch.floor(item / 15)).flatten().to(torch.int)] += 1
            x_quant += -res * 15
            # x_quant += -y
            # ListCount[(torch.floor(y / 15)).flatten().to(torch.int)] += 1
            y = F.linear(x_linear[:, (i * 4):min((i + 1) * 4, x_linear.size(1))], q_minus_high[:, (i * 4):min((i + 1) * 4, x_linear.size(1))])
            index = torch.randint(0, 1000, y.size())
            res = ProbSample[y.to(torch.int), index] + y / 15 - (y / 15).detach()
            x_quant += -2 * res * 15
            # x_quant += -2 * y
            # for item in y.flatten():
            #     ListCount[(torch.floor(item / 15)).flatten().to(torch.int)] += 1
        # x_plus = F.linear(x_linear, q_plus, bias=None)
        # x_minus_low = F.linear(x_linear, q_minus_low, bias=None)
        # x_minus_high = F.linear(x_linear, q_minus_high, bias=None)
        # x_quant = x_plus - x_minus_low - 2 * x_minus_high
        # x_quant = F.linear(x_linear, q_quant, bias=None)
        x_quant = x_quant.view(x.size(0), -1, x.size(1))
        x_quant = x_quant.transpose(1, 2)
        x_quant = x_quant - bias
        x_quant = x_quant / 16
        return x_quant
    
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
        # print("Input", np.unique(temp.flatten()))
        q_w = FakeQuantize.apply(self.conv_module.weight, self.qw)
        # temp = q_w.cpu().detach().numpy()
        # print("Weight", np.unique(temp.flatten()))
        x_quant = self.hardware_action(x, q_w, self.q_w_plus, self.q_w_minus_low, self.q_w_minus_high)
        x = self.fake_o.apply(x_quant, self.qo)

        return x


class QLinear(nn.Module):
    def __init__(self, fc_module, Qn, num_bits, Qn_io, num_bits_io) -> None:
        super().__init__()
        self.fc_module = fc_module
        self.num_bits = num_bits
        self.Qn = Qn
        self.Qn_io = Qn_io
        self.num_bits_io = num_bits_io
        self.qw = QParam(self.Qn, self.num_bits)  ## 卷积层权重量化
        self.qi = QParam(self.Qn_io, self.num_bits_io)  ## 卷积层输入量化
        self.qo = QParam(2, 6)  ## 卷积层输出量化
        self.fake_i = FakeQuantize()
        self.fake_o = FakeQuantize()

    def forward(self, x):
        x = self.fake_i.apply(x, self.qi)
        q_w = FakeQuantize.apply(self.fc_module.weight, self.qw)
        x = F.linear(x, q_w)
        x = self.fake_o.apply(x, self.qo)
        return x
    
    def QGinfer(self, x, mean, std, invert=True):
        x = self.fake_i.apply(x, self.qi)
        q_w = FakeQuantize.apply(self.fc_module.weight, self.qw)
        if invert:
            q_g_w = invert_gaussian(q_w, mean, std)
        else:
            q_g_w = q_w + gaussian_noise(q_w, mean, std)
        x = F.linear(x, q_g_w)
        x = self.fake_o.apply(x, self.qo)
        return x

    def freeze(self):
        self.fc_module.weight.data = self.qw.quantize_tensor(self.fc_module.weight.data)
        self.fc_module.weight.data = self.qw.dequantize_tensor(self.fc_module.weight.data)

    def quantize_gaussian_inference(self, x, mean, std, invert=True):
        # temp = x.cpu().detach().numpy()
        # np.save('D:/PycharmProjects/1d_cnn_fixpoint/1D_CNN_dataset/chb01/test0/chb01_03_0_s_fake_int.npy',temp)

        ## use QG infer
        x = self.QGinfer(x, mean, std, invert)

        return x
    
    def quantize_inference(self, x):
        x = self.fake_i.apply(x, self.qi)
        x = F.linear(x, FakeQuantize.apply(self.fc_module.weight, self.qw))
        x = self.fake_o.apply(x, self.qo)
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
        x = F.avg_pool1d(x, self.kernel_size, self.stride, self.padding)
        return x

    def quantize_inference(self, x):
        return F.avg_pool1d(x, self.kernel_size, self.stride, self.padding)


class Net(nn.Module):
    def __init__(self, ):
        super(Net, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=18, out_channels=4, kernel_size=7, bias=False)
        self.linear = nn.Linear(in_features=4, out_features=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        print(x.size())
        x = self.conv1(x)
        print(x.size())
        x = F.relu(x)
        print(x.size())
        x = F.avg_pool1d(x, 26)
        print(x.size())
        x = x.view(-1, 4)
        print(x.size())
        x = self.linear(x)
        print(x.size())
        return x

    def quantize(self, Qn, num_bits, Qn_io, num_bits_io):
        self.qconv1 = QConv1d(self.conv1, Qn, num_bits, Qn_io, num_bits_io)
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
        x = self.qfc.quantize_gaussian_inference(x, mean, std, invert)
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
