from main import Communication_Init, Communication_Write, Communication_Read, HandShake, Write, GroupConfig
from aardvark_api.python.aardvark_py import *
import numpy as np
import time


def Dist_Comp(AvailablePlace, StorePlace, SampleNum=1000):
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
    # Data Generation
    Data = np.zeros((61, SampleNum, Parallelism))
    for i in range(61):
        UpperBound = min(i, 15) + 1
        LowerBound = max(0, i - 45)
        for j in range(SampleNum):
            SampleList = []
            UpperLimit = UpperBound
            LowerLimit = LowerBound
            Sum = 0
            for k in range(Parallelism - 1):
                temp = np.random.randint(LowerLimit, UpperLimit)
                SampleList.append(temp)
                Sum += temp
                UpperLimit = min(i - Sum, 15) + 1
                LowerLimit = max(0, i - Sum - 15 * (2 - k))
            SampleList.append(i - Sum)
            np.random.shuffle(SampleList)
            Data[i, j, :] = np.array(SampleList)
    Result_Batch = np.zeros((61, SampleNum))
    ratio = 0
    monument = 0.01
    count = 1
    for i in range(61):
        data_LIST = []
        data_TEMP = 0
        # Write In Weight
        ratio = i / 60
        if ratio > monument:
            monument = min(ratio + monument, 1)
        print(i, '/', 60)
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
        for j in range(SampleNum):
            # Write In Data
            if count % 300 == 0:
                time.sleep(1)
            data_LIST = []
            IndexString = ""
            Test_Result = 0
            for k in range(Parallelism):
                data_TEMP <<= DATA_BIT
                Test_Result += Data[i, j, k]
                data_TEMP += int(Data[i, j, k])
                if k % 2 == 1:
                    data_LIST.append(data_TEMP)
                    if Test_Result > 15:
                        Zero = False
                    data_TEMP = 0
            if Zero:
                continue
            count = count + 1
            Zero = True
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
            # Read Out Result
            NotSuccess = Communication_Write(handle=handle, data=result_addr, DEVICE=DEVICE_COMM, mode=AA_I2C_NO_STOP)
            NotSuccess = Communication_Read(handle=handle, data=readin, DEVICE=DEVICE_COMM, mode=AA_I2C_NO_FLAGS)
            print(readin[0])
            Result_Batch[i, j] = min(3, readin[0])
            Test_Result = 0
    aa_close(handle)
    return Result_Batch


def main():
    AvailablePlace = [2, 3, 3]
    StorePlace = [7, 1, 2, 1]
    # return
    SampleNum = 1
    CompResult = Dist_Comp(AvailablePlace, StorePlace, SampleNum)
    Dist = np.zeros((61, 4))
    for i in range(61):
        for j in range(SampleNum):
            Dist[i, int(CompResult[i, j])] += 1
    Dist /= SampleNum
    File = open("DeviceDist.txt", "w")
    print("Prob = [", end='', file=File)
    for i in range(61):
        print("[", end='', file=File)
        for j in range(4):
            print(Dist[i, j], end='', file=File)
            if j != 3:
                print(", ", end='', file=File)
        print("]", end='', file=File)
        if i != 60:
            print(",\n\t\t", end='', file=File)
        else:
            print("]", end='', file=File)
    return


if __name__ == "__main__":
    main()