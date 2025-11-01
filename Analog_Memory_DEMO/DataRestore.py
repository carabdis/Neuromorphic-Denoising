import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def main():
    content = pd.read_excel("C:/Users/16432/Desktop/Workplace/Analog_Memory/Test_Result_Golden.xlsx", sheet_name="PCB_PULSE_WRITE")
    content = content.to_numpy()
    test_device = content[74:, :5]
    Reset_Dict = dict()
    NameList = []
    for message in test_device:
        [Type, Voltage, Time, _, Res] = message
        if Type == "Reset" and Reset_Dict.get(Voltage) == None:
            Reset_Dict[Voltage] = []
            Reset_Dict[Voltage].append(Res)
            NameList.append(Voltage)
        elif Type == "Reset":
            Reset_Dict[Voltage].append(Res)
    plt.figure()
    plt.semilogx()
    plt.hist(Reset_Dict[2.7], alpha=0.5)
    # plt.hist(Reset_Dict[2.6], alpha=0.5)
    plt.hist(Reset_Dict[2.5], alpha=0.5)
    # plt.hist(Reset_Dict[2.4], alpha=0.5)
    plt.hist(Reset_Dict[2.3], alpha=0.5)
    # for name in NameList:
    #     plt.hist(Reset_Dict[2.])
    plt.show()
    return

if __name__ == "__main__":
    main()