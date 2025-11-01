import torch
from torch import nn
import h5py as hpy
import numpy as np
from pytorch_model import Sign, Round
import pytorch_model as model
from pytorch_model import ProbSample
from main import DeployWeightLog, Communication_Init, Communication_Write, Communication_Read, HandShake, Write, GroupConfig
from sklearn.model_selection import train_test_split
from pytorch_utils import noise, display
from aardvark_api.python.aardvark_py import *
import matplotlib.pyplot as plt
from PIL import Image
import time


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def loss_function(x_static, x_hat_static, mean_static, log_var_static):
    # reproduction_loss = torch.mean(torch.sum(nn.functional.l1_loss(x_static, x_hat_static, reduction="none"), dim=(2, 3)))
    x_hat_static = torch.clamp(x_hat_static, 0, 1)
    reproduction_loss = nn.functional.binary_cross_entropy(x_hat_static, x_static, reduction='sum')
    KLD = - 0.5 * torch.sum(1 + log_var_static - mean_static.pow(2) - log_var_static.exp())

    return reproduction_loss + KLD


def Parallel_Compute_Serial_Ctrl(Data, AvailablePlace, Weight, model, StorePlace, Initial_Weight=[1, 1, 1, 1], GROUP_TOTAL=12, Result_Dim=32):
    WL, BL, COL = AvailablePlace
    WL_SUB, BL_SUB, COL_SUB, ROW_SUB = StorePlace
    COL_BIT = 3
    DATA_BIT = 4
    GROUP = 4
    Parallelism = 4
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
    Data = torch.clip(Round.apply((Data - model.encoder.conv1.InMin) / (model.encoder.conv1.InMax - model.encoder.conv1.InMin) * 15), 0, 15).numpy().astype(int)
    Result_Batch = np.zeros((Data.shape[0], Result_Dim))
    Test_Result = 0
    Sign = 1
    ratio = 0
    monument = 0.01
    target = monument
    CompDict = dict()
    RepeatDict = dict()
    count = 1
    for i in range(int(len(Weight) / Parallelism)):
        data_LIST = []
        data_TEMP = 0
        # Write In Weight
        ratio = i / (int(len(Weight) / Parallelism))
        if ratio > target:
            target = min(ratio + monument, 1)
            print(i, '/', int(len(Weight) / Parallelism))
        for j in range(Parallelism):
            if Weight[Parallelism * i + j] == Initial_Weight[j]:
                continue
            else:
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
        for j in range(Data.shape[0]):
            if count % 300 == 0:
                time.sleep(10)
                count = 1
            # Write In Data
            data_LIST = []
            IndexString = ""
            for k in range(Parallelism):
                data_TEMP <<= DATA_BIT
                Test_Result += Data[j, Index * Parallelism + k] * Weight[Parallelism * i + k]
                IndexString += str(Data[j, Index * Parallelism + k]) + str(Weight[Parallelism * i + k])
                data_TEMP += int(Data[j, Index * Parallelism + k] * Weight[Parallelism * i + k])
                if k % 2 == 1:
                    data_LIST.append(data_TEMP)
                    data_TEMP = 0
            count = count + 1
            for k in range(len(data_LIST)):
                data_COMP = array('B', [COMPUTE_BYTE + int(k), abs(data_LIST[k])])
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
            NotSuccess = Communication_Write(handle=handle, data=result_addr, DEVICE=DEVICE_COMM, mode=AA_I2C_NO_STOP)
            NotSuccess = Communication_Read(handle=handle, data=readin, DEVICE=DEVICE_COMM, mode=AA_I2C_NO_FLAGS)
            Result_Batch[j, int(i / GROUP_TOTAL / 2)] += Sign * min(3, readin[0])
            CompDict[IndexString] = min(3, readin[0])
            RepeatDict[IndexString] = 0
            Test_Result = 0
        if i % GROUP_TOTAL == GROUP_TOTAL - 1:
            Sign = -Sign
    aa_close(handle)
    return Result_Batch


def VAE_DEMO():
    model = torch.load("./Analog_Memory_DEMO/MyModel/checkpoint_3425.ptr", map_location=torch.device("cpu"))
    w = torch.reshape(model.encoder.conv1.layer.weight, (model.encoder.conv1.layer.weight.size(0), -1))
    weight = Sign.apply(w).detach().numpy().transpose()
    DeployList = np.zeros((weight.shape[0], 2 * weight.shape[1]))
    weight = weight.transpose()
    for i in range(weight.shape[1]):
        for j in range(weight.shape[0]):
            if weight[j, i] == 1:
                DeployList[i][2 * j] = 1
            elif weight[j, i] == -1:
                DeployList[i][2 * j + 1] = 1
    DeployWeightLog(DeployList)
    WeightList = []
    DataChannel = DeployList.shape[0]
    DeployChannel = DeployList.shape[1]
    for i in range(DeployChannel):
        for j in range(DataChannel):
            WeightList.append(DeployList[j][i])
    dataset = hpy.File("./Analog_Memory_DEMO/data/3dshapes.h5", "r")
    dataset_images = dataset["images"]
    dataset_labels = dataset["labels"]
    IndexFile = open("./Analog_Memory_DEMO/IndexFile.txt", "r")
    Index = IndexFile.read().split('\n')[:-1]
    Index = [int(i) for i in Index]
    SampleNum = 1
    n = 0
    while n < SampleNum:
        Index = 384502
        train_images, test_images, train_labels, test_labels = train_test_split(dataset_images[int(Index):int(Index) + 20, :, :, :], dataset_labels[:20, :], random_state=42)
        test_ds = test_images[0:1, :, :, :]
        np.random.shuffle(test_ds)
        train_images = torch.tensor(np.transpose(train_images, (0, 3, 1, 2))).float()
        noisy_test_numpy = noise(test_ds / 255., noise_factor=0.1)
        test_tensor = torch.tensor(np.transpose(test_ds / 255, (0, 3, 1, 2))).float()
        noisy_test = torch.tensor(np.transpose(noisy_test_numpy, (0, 3, 1, 2))).float()
        origin, decoded_imgs, mean, var = model(noisy_test)
        loss_ori = loss_function(test_tensor, decoded_imgs, mean, var)
        if loss_ori > 3200:
            continue
        print(Index)
        n = n + 1
        decoded_imgs_numpy = np.transpose(decoded_imgs.detach().numpy(), (0, 2, 3, 1))
        AvailablePlace = [2, 3, 3]
        StorePlace = [7, 1, 2, 1]
        with torch.no_grad():
            BatchData = model.encoder.Drop(noisy_test)
            BatchData = nn.functional.unfold(BatchData, model.encoder.conv1.layer.kernel_size, model.encoder.conv1.layer.dilation, model.encoder.conv1.layer.padding, 
                                            model.encoder.conv1.layer.stride).transpose(1, 2)
            Batch_Size = BatchData.size(0)
            BatchData = BatchData.reshape((-1, BatchData.size(2)))
            Index = np.arange(BatchData.size(0))
            np.random.shuffle(Index)
            HardwareLength = int(0.01 * BatchData.size(0))
            Index = Index[:HardwareLength]
            HardwareResult = Parallel_Compute_Serial_Ctrl(BatchData[Index], AvailablePlace, WeightList, model, StorePlace)
            HardwareResult = torch.tensor(HardwareResult)
            HardwareResult = HardwareResult.float()
            BatchData = torch.clip(Round.apply((BatchData - model.encoder.conv1.InMin) / (model.encoder.conv1.InMax - model.encoder.conv1.InMin) * 15), 0, 15)
            Min = torch.ones(BatchData.size()) * model.encoder.conv1.InMin
            OutList = []
            w = torch.reshape(model.encoder.conv1.layer.weight, (model.encoder.conv1.layer.weight.size(0), -1))
            w = Sign.apply(w)
            bias = torch.nn.functional.linear(Min, w)
            wPlus = model.encoder.conv1.m(w)
            wMinus = model.encoder.conv1.m(-w)
            x = torch.zeros((BatchData.size()[0], w.size()[0]), device=DEVICE)
            for i in range(int(np.ceil(BatchData.size()[1] / 4))):
                y = torch.nn.functional.linear(BatchData[:, (i * 4):(i + 1) * 4], wPlus[:, (i * 4):(i + 1) * 4])
                OutList.append(y)
                y = torch.nn.functional.linear(BatchData[:, (i * 4):(i + 1) * 4], wMinus[:, (i * 4):(i + 1) * 4])
                OutList.append(-y)
            for i in range(len(OutList)):
                tensor = OutList[i]
                LongTensor = torch.clip(Round.apply(tensor), -60, 60).long()
                QuantTensor = torch.clip(Round.apply(tensor / 15), -3, 3)
                RandTensor = torch.randint(0, 1000, QuantTensor.size(), device=DEVICE)
                TempTensor = torch.reshape(ProbSample[torch.abs(LongTensor.flatten()), RandTensor.flatten()],
                                        QuantTensor.size())
                AddTensor = TempTensor * torch.sign(QuantTensor) + QuantTensor - QuantTensor.detach()
                x = (x + AddTensor * (model.encoder.conv1.InMax - model.encoder.conv1.InMin))
            x[Index] = HardwareResult
            x = x + bias + model.encoder.conv1.layer.bias
            x = x.reshape((Batch_Size, -1, x.size(1)))
            dim_value = int(np.sqrt(x.size(1)))
            x = x.transpose(1, 2).reshape(x.size(0), x.size(2), dim_value, dim_value)
            x = x.float()
            x = model.encoder.bn1(x)
            x = model.encoder.act1(x)

            x = model.encoder.conv2(x)
            x = model.encoder.bn2(x)
            x = model.encoder.act2(x)

            x = model.encoder.conv3(x)
            x = model.encoder.bn3(x)
            x = model.encoder.act3(x)

            x = model.encoder.conv4(x)
            x = model.encoder.bn4(x)
            x = model.encoder.act4(x)

            x = model.encoder.pool(x)
            x = torch.flatten(x, start_dim=1)

            x_mean = model.encoder.mean_linear(x)
            x_var = model.encoder.var_linear(x)
            latent = model.encoder.sample(x_mean, x_var)
            x_imp = model.decoder(latent)
            loss = loss_function(test_tensor, x_imp, x_mean, x_var)
        hardware_imgs_numpy = np.transpose(x_imp.detach().numpy(), (0, 2, 3, 1))
        File = open("./Analog_Memory_DEMO/Results/Loss.txt", "a")
        print("loss" + str(n) + ":", loss_ori, loss)
        print("loss" + str(n) + ":", loss_ori, loss, file=File)
        img = Image.open("./Analog_Memory_DEMO/Results/testfig1.png")
        plt.figure(figsize=(12, 6))
        plt.imshow(img)
        plt.axis("off")
        plt.show()
        # fig = display(hardware_imgs_numpy, decoded_imgs_numpy, title='Inputs and outputs for VAE', n=1, num=n)
    return


def main():
    VAE_DEMO()
    return


if __name__ == "__main__":
    main()

# figA: 3694.5930 v.s. 3468.4443
# figB: 3036.1611 v.s. 3408.8677
# figC: 7557.3286 v.s. 7969.8306
# figD: 7921.8931 v.s. 7111.6685
# figE: 4583.3516 v.s. 6457.7866
# figF: 3075.2993 v.s. 4092.1855
# figG: 2878.4973 v.s. 5327.9927
# figH: 3195.3872 v.s. 3564.9749
# figI: 3020.1934 v.s. 2865.4031
# figJ: 2912.4709 v.s. 3502.6064
# figK: 3015.0107 v.s. 2815.2805
# figL: 3017.1868 v.s. 3225.5444
# figM: 3070.2454 v.s. 2934.4111
# figN: 3074.5007 v.s. 3084.5920
# figO: 2918.2769 v.s. 2954.8096