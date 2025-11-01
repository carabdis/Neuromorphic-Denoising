'''
Author: CSuperlei
Date: 2022-10-03 10:16:14
LastEditTime: 2022-10-10 20:35:35
Description: 
'''
from json import load
import torch
import numpy as np
from model_fixed import Net
import matplotlib.pyplot as plt

if __name__ == '__main__':
    device = 'cuda'
    model = Net().to(device)
    print(model)
    load_quant_model_file = 'D:/PKU/1d_cnn_fixpoint-foryuqi/1d_cnn_fixpoint-foryuqi/ckpt/1d_cnn_fix_4bit.pt'
    if load_quant_model_file is not None:
        weight = torch.load(load_quant_model_file, map_location='cuda')

        for k, v in weight.items():
            print(k, v.shape)

            temp = v.cpu().detach().numpy()
            temp = temp.reshape(-1)
            x_axis = np.arange(0, temp.shape[0])
            plt.scatter(x_axis, temp)
            plt.title(k)
            plt.show()
        ## 无法完全加载是因为model.load_state_dict()只加载和模型有关的参数,即模型__init__()定义的内容
        # model.load_state_dict(torch.load(load_quant_model_file, map_location='cuda'), strict=False) 
        print(model.state_dict())
        print("Successfully load quantized model %s" % load_quant_model_file)

    '''
    temp = np.load('D:/PycharmProjects/1d_cnn_fixpoint/1D_CNN_dataset/chb01/test0/chb01_03_0_s_fake_int.npy')
    temp = temp.reshape(-1)
    x_axis = np.arange(temp.shape[0])
    plt.scatter(x_axis, temp)
    plt.title('input_fake_int')
    plt.show()
    '''

    model.load_state_dict(torch.load('ckpt/1d_cnn.pt', map_location='cpu'))
    for layer, param in model.state_dict().items():  # param is weight or bias(Tensor)
        print(layer, param)
        temp = param.view(-1)
        temp = temp.cpu().detach().numpy()
        x_axis = np.arange(temp.shape[0])
        plt.scatter(x_axis, temp)
        plt.title(layer)
        plt.show()