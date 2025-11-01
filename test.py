# import galois
# import torch
import os

from torch import nn
from torch import quantization as q
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import transforms
import matplotlib as mpl
import scipy
import numpy as np
import galois
from pathlib import Path
from distfit import distfit
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42


def scatter_line(data, scale=0, error_kind="se"):
    AnalogUpperLim = 4
    AnalogLowerLim = 0.1
    RefValue = np.log10(AnalogUpperLim / AnalogLowerLim)
    AnalogError = []
    for index, row in data.iterrows():
        AnalogError.append(np.clip(np.log10(row["Read Current"] / AnalogLowerLim),
                                   0, RefValue) / RefValue * 25.2)
    data.insert(4, "Analog Error", AnalogError, True)
    sns.set_theme(style="darkgrid")
    # sns.pointplot(data=data, x="Value", y="Write Voltage", errorbar=(error_kind, scale))
    # sns.scatterplot(data=data, x="Value", y="Write Voltage",)
    p = sns.regplot(data=data, x="Value", y="Write Voltage")
    slope, intercept, r, p, sterr = scipy.stats.linregress(x=p.get_lines()[0].get_xdata(),
                                                           y=p.get_lines()[0].get_ydata())
    plt.text(40, 1, 'y = ' + str(round(intercept, 3)) + ' + ' + str(round(slope, 3)) + 'x')
    r, p = scipy.stats.pearsonr(data['Write Voltage'], data['Value'])
    plt.text(40, 0.75, 'r={:.2f}, p={:.2g}'.format(r, p))
    # ax = plt.twinx()
    # sns.scatterplot(data=data, x="Value", y="Pulse Width", ax=ax, color="orange")
    # sns.scatterplot(data=data, x="Value", y="Analog Error", ax=ax, color="green")
    # plt.show()
    plt.savefig("Voltage.pdf")
    return


def density(data):
    total = 60
    step = 5
    group = int(total / step)
    IndexDict = np.array([int(8.4 * i) for i in range(4)])
    DownLimit = [i * step for i in range(group)]
    UpLimit = [i * step + step - 1 for i in range(group)]
    UpLimit[-1] = total
    PassString = [str(DownLimit[i]) + '-' + str(UpLimit[i]) for i in range(group)]
    PulseDict = [str(8.4 * i) for i in range(4)]
    GroupData = np.zeros((data.shape[0], 2)).astype(float)
    Count = 0
    for index, row in data.iterrows():
        Group = np.clip(int(row["Value"] / step), 0, group - 1)
        # OutGroup = np.where(IndexDict == int(row["Pulse Width"]))[0]
        # print(Group, OutGroup)
        GroupData[Count, 0] = DownLimit[Group]
        GroupData[Count, 1] = row["Pulse Width"]
        Count = Count + 1
    GroupData = pd.DataFrame(GroupData).rename(columns={0: "Value", 1: "Pulse Width"})
    # calculate the distribution of `Clicked` per `Rank`
    distribution = pd.crosstab(GroupData["Value"], GroupData["Pulse Width"], normalize='index')
    print(distribution)
    # print(distribution)

    # print(distribution)

    # plot the cumsum, with reverse hue order
    sns.barplot(data=distribution.cumsum(axis=1).stack().reset_index(name='Dist'),
                x='Value', y='Dist', hue='Pulse Width',
                hue_order=distribution.columns[::-1],  # reverse hue order so that the taller bars got plotted first
                dodge=False)
    plt.show()
    plt.scatter(data['Write Voltage'], data["Read Current"])
    plt.semilogy()
    plt.xlim(1)
    plt.yticks(fontproperties='Arial')
    plt.xticks(fontproperties='Arial')
    plt.savefig("WriteVoltage.pdf")
    plt.show()
    return


def DNL(data):
    Group = 4
    TotalValue = 60
    Step = int(TotalValue / Group)
    DiffStep = 1
    DiffGroup = int(TotalValue / DiffStep)
    AnalogUpperLim = 4
    AnalogLowerLim = 0.1
    RefValue = np.log10(AnalogUpperLim / AnalogLowerLim)
    PulseDict = [8.4 * (Group - i - 1) for i in range(Group)]
    DownLimit = [DiffStep * i for i in range(DiffGroup)]
    IdealValue = []
    ErrorValue = []
    DownLim = []
    AnalogError = []
    for index, row in data.iterrows():
        Ideal = PulseDict[np.clip(int(row["Value"] / Step), 0, Group - 1)]
        IdealValue.append(Ideal)
        ErrorValue.append(int((row["Pulse Width"] - Ideal) / 8.4))
        DownLim.append(DownLimit[np.clip(int(row["Value"] / DiffStep), 0, DiffGroup - 1)])
        AnalogError.append((np.clip(np.log10(row["Read Current"] / AnalogLowerLim),
                                    0, RefValue) / RefValue * 60 - 60 + row["Value"]) / 15)
    # IdealValue = np.array(IdealValue).reshape((len(IdealValue), 1))
    # print(IdealValue)
    data.insert(1, "Ideal Value", IdealValue, True)
    data.insert(5, "Error Value", ErrorValue, True)
    data.insert(6, "Value Group", DownLim, True)
    data.insert(7, "Analog Error", AnalogError, True)
    sns.set_theme(style="whitegrid")
    IntError = [0, 0]
    for index, row in data.iterrows():
        # print(row["Value"], row["Error Value"], row["Analog Error"])
        IntError[0] += abs(row["Error Value"])
        IntError[1] += abs(row["Analog Error"])
    # print(IntError)
    value = np.mean(data["Value"].to_numpy())   
    error = np.mean(abs(data["Error Value"].to_numpy()) * 15)
    AnaError = np.mean(abs(data["Analog Error"].to_numpy()) * 15)
    print(np.log10(value / AnaError) * 20, np.log10(value / error) * 20)
    print(np.log10(value / AnaError) * 20 - np.log10(value / error) * 20)
    return
    sns.scatterplot(
        data=data, x="Value", y="Error Value",
        color="orange"
    )
    sns.scatterplot(
        data=data, x="Value", y="Analog Error",
    )
    # plt.savefig("Residual.pdf")
    plt.show()
    plt.bar(["Error", "Raw"], IntError)
    # plt.savefig("TotalError.pdf")
    plt.show()
    return


def get_index(Name):
    Name = Name.split("_")
    return int(Name[1])


def get_point(Name):
    Name = Name.split("_")
    PointWithSuffix = Name[-1]
    Point = PointWithSuffix[:-4]
    return float(Point)


def DrawPicture(Direct):
    FileList = os.listdir(Direct)
    DeployFile = []
    OriginFile = []
    DeployPoint = 0
    OriginPoint = 0
    for name in FileList:
        if "deployed" in name:
            DeployFile.append(name)
            DeployPoint += get_point(name)
        elif "origin" in name:
            OriginFile.append(name)
            OriginPoint += get_point(name)
    OriginFile.sort(key=get_index)
    DeployFile.sort(key=get_index)
    DeployPoint /= len(DeployFile)
    OriginPoint /= len(OriginFile)
    # print(DeployPoint, OriginPoint)
    DeployArray = [plt.imread(Direct + "/" + name) for name in DeployFile]
    OriginArray = [plt.imread(Direct + "/" + name) for name in OriginFile]
    DeployRow = []
    OriginRow = []
    TempGroupA = []
    TempGroupB = []
    for i in range(len(DeployFile)):
        TempGroupA.append(DeployArray[i])
        TempGroupB.append(OriginArray[i])
        if i % 5 == 4:
            TempGroupA = np.concatenate(TempGroupA, axis=1)
            TempGroupB = np.concatenate(TempGroupB, axis=1)
            DeployRow.append(TempGroupA)
            OriginRow.append(TempGroupB)
            TempGroupA = []
            TempGroupB = []
    FinalFigA = np.concatenate(DeployRow, axis=0)
    FinalFigB = np.concatenate(OriginRow, axis=0)
    plt.imshow(FinalFigA)
    plt.savefig(Direct + "/Hardware.pdf")
    # plt.show()
    plt.imshow(FinalFigB)
    plt.savefig(Direct + "/Software.pdf")
    # plt.show()
    return


def WeightMap():
    WeightRaw = pd.read_excel("VAE_weight_map.xlsx", header=None)
    WeightCNN = pd.read_csv("DeployWeight.csv", header=None)
    WeightRaw = WeightRaw.to_numpy()
    WeightInit = WeightRaw[0:16]
    WeightHard = WeightRaw[17:33]
    WeightInit = np.vstack((WeightInit[:, 0:32], WeightInit[:, 32:64]))
    WeightHard = np.vstack((WeightHard[:, 0:32], WeightHard[:, 32:64]))
    fig, axes = plt.subplots(2, 1)
    images = [axes[0].imshow(WeightInit), axes[1].imshow(WeightHard)]
    fig.colorbar(images[0], ax=axes, orientation='vertical', fraction=.1)
    plt.savefig("WeightMap_VAE.pdf")
    plt.show()
    # Plus = WeightHard[np.where(WeightHard > 2)]
    # Zero = WeightHard[np.where(WeightHard < 2)]
    # Average = [np.average(Plus), np.average(Zero)]
    # Sigma = [np.std(Plus), np.std(Zero)]
    # WeightCNN = WeightCNN.to_numpy()
    # WeightCNNDeploy = np.zeros(WeightCNN.shape)
    # RandomDeploy = [np.random.normal(Average[0], Sigma[0], WeightCNNDeploy.shape),
    #                 np.random.normal(Average[1], Sigma[1], WeightCNNDeploy.shape)]
    # WeightCNNDeploy[np.where(WeightCNN > 0.5)] = RandomDeploy[0][np.where(WeightCNN > 0.5)]
    # WeightCNNDeploy[np.where(WeightCNN < 0.5)] = np.clip(RandomDeploy[1][np.where(WeightCNN < 0.5)], 0, 1)
    # fig, axes = plt.subplots(2, 1)
    # images = [axes[0].imshow(WeightCNN), axes[1].imshow(WeightCNNDeploy)]
    # fig.colorbar(images[0], ax=axes, orientation='vertical', fraction=.1)
    # plt.savefig("WeightMap_CNN.pdf")
    # plt.show()
    return


def AppliDraw():
    HardwareVAE = 150.09449795918368
    SoftwareVAE = 131.5541346938775
    HardwareCNN_acc = ((2879 - 32) * 31/32 + 24 + 4) / 2879
    SoftwareCNN_acc = 0.985411
    print(HardwareVAE, SoftwareVAE, (SoftwareVAE - HardwareVAE) / HardwareVAE)
    print(HardwareCNN_acc, SoftwareCNN_acc, (SoftwareCNN_acc - HardwareCNN_acc) / SoftwareCNN_acc)
    HardwareCNN_sen = 24/28
    SoftwareCNN_sen = 20/32
    plt.bar(["HardwareVAE", "SoftwareVAE"], [HardwareVAE, SoftwareVAE])
    # plt.ylim(120)
    # plt.savefig("VAE_perf.pdf")
    plt.show()
    plt.bar(["HardwareCNN_acc", 'SoftwareCNN_acc'], [HardwareCNN_acc, SoftwareCNN_acc])
    # plt.ylim(0.9)
    # plt.savefig("CNN_ACC.pdf")
    plt.show()
    # plt.bar(["HardwareCNN_sen", 'SoftwareCNN_sen'], [HardwareCNN_sen, SoftwareCNN_sen])
    # plt.ylim(0.5)
    # plt.savefig("CNN_SEN.pdf")
    # plt.show()
    return


def CellDraw():
    # plt.bar(["FSD", "ISSCC 2023", "Eyeriss", "ISSCC 2022", "VLSI 2022", "Ours"],
    #         [1.152, 6.002, 8.337, 31.04, 35.23, 62.58])
    plt.bar(["True\nNorth", "Loihi", "Tianjic", "Ours"],
            [0.4, 0.106, 1.27, 30.79])
    # plt.bar(["Neuro-CIM", "NeuRRAM", "Ours"],
    #         [413.8, 317.59, 400.5])
    # area efficiency
    Software = [3694.5930, 3036.1611, 3075.2993, 3195.3872, 3020.1934, 2912.4709,
                3015.0107, 3017.1868, 3070.2454, 3074.5007, 2918.2769]
    Hardware = [3468.4443, 3408.8677, 4092.1855, 3564.9749, 2865.4031, 3502.6064,
                2815.2805, 3225.5444, 2934.4111, 3084.5920, 2954.8096]
    Soft = np.mean(Software)
    Hard = np.mean(Hardware)
    # plt.bar(["Software", "Hardware"],
    #         [Soft, Hard])
    print(Soft, Hard, (Soft - Hard) / Hard)
    # plt.savefig("Accuracy.pdf")
    # plt.show()
    # plt.bar(["ADC Free", "ARCHON", "NeuRRAM", "Ours"],
    #         [65.60, 289, 470.38, 1468.89])
    # plt.semilogy()
    print(30.79 / 1.27, 1468.89 / 470.38)
    plt.savefig("Efficiency_digital.pdf")
    # plt.show()
    return


def VoltageCurrent(data, source, drain):
    AnalogUpperLim = 4
    AnalogLowerLim = 0.1
    RefValue = np.log10(AnalogUpperLim / AnalogLowerLim)
    AnalogError = []
    for index, row in data.iterrows():
        AnalogError.append(np.clip(np.log10(row["Read Current"] / AnalogLowerLim),
                                   0, RefValue) / RefValue * 25.2)
    data.insert(4, "Analog Error", AnalogError, True)
    sns.set_theme(style="darkgrid")
    plt.figure(figsize=(6, 8))
    # sns.pointplot(data=data, x="Value", y="Write Voltage", errorbar=(error_kind, scale))
    # sns.scatterplot(data=data, x="Value", y="Write Voltage",)
    p = sns.regplot(data=data, x="Read Current", y="Write Voltage", logx=True)
    slope, intercept, r, p, sterr = scipy.stats.linregress(x=p.get_lines()[0].get_xdata(),
                                                           y=p.get_lines()[0].get_ydata())
    # plt.text(40, 1, 'y = ' + str(round(intercept, 3)) + ' + ' + str(round(slope, 3)) + 'x')
    print('y = ' + str(round(intercept, 3)) + ' + ' + str(round(slope, 3)) + 'x')
    r, p = scipy.stats.pearsonr(data['Write Voltage'], data['Read Current'])
    # plt.text(40, 0.75, 'r={:.2f}, p={:.2g}'.format(r, p))
    SimPoint = np.average(data["Write Voltage"].to_numpy())
    print(SimPoint, np.exp(-intercept/slope))
    AverageCurrent = np.average(data["Read Current"].to_numpy())
    WriteNoise = pd.read_csv(source)
    ReadNoise = np.zeros(WriteNoise.shape)
    for index, row in WriteNoise.iterrows():
        ReadNoise[index, 0] = row["output noise; V / sqrt(Hz) X"]
        ReadNoise[index, 1] = (np.exp((SimPoint - intercept) / slope) - np.exp((row["output noise; V / sqrt(Hz) Y"] + SimPoint - intercept) / slope)) * 0.2 / AverageCurrent
    ReadNoise = pd.DataFrame(ReadNoise,columns=["output noise; V / sqrt(Hz) X", "output noise; V / sqrt(Hz) Y"])
    ReadNoise.to_csv(drain, sep=' ', index=None)
    plt.semilogx()
    # plt.gca().invert_xaxis()
    # plt.gca().invert_yaxis()
    plt.show()
    # plt.savefig("Write&Read.pdf")
    return


def VoltageTimeVoltage(data, source, drain):
    AnalogUpperLim = 4
    AnalogLowerLim = 0.1
    Step = (AnalogUpperLim / AnalogLowerLim) ** (1/3)
    RefValue = np.log(AnalogUpperLim / AnalogLowerLim)
    VoltageRange = [0.2755, 0.38589]
    CurrentRange = [np.min(data["Read Current"].to_numpy()), np.max(data["Read Current"].to_numpy())]
    Ratio = (CurrentRange[1] - CurrentRange[0]) / (VoltageRange[1] - VoltageRange[0])
    WriteNoise = pd.read_csv(source)
    ReadNoise = np.zeros(WriteNoise.shape)
    for index, row in WriteNoise.iterrows():
        ReadNoise[index, 0] = row["output noise; V / sqrt(Hz) X"]
        Current = row["output noise; V / sqrt(Hz) Y"] * Ratio
        ReadNoise[index, 1] = (np.log10((Current + AnalogLowerLim * Step) / (AnalogLowerLim * Step)) ) / RefValue * 25.2 * 0.2
    ReadNoise = pd.DataFrame(ReadNoise,columns=["output noise; V / sqrt(Hz) X", "output noise; V / sqrt(Hz) Y"])
    ReadNoise.to_csv(drain, sep=' ', index=None)
    return


def main():
    data = pd.read_csv("Data.csv")
    # kind = np.array([["Write Voltage", "Pulse Width"] for i in range(data.shape[0])])
    # Voltage = np.vstack((data[:, 0], data[:, 1], kind[:, 0]))
    # Width = np.vstack((data[:, 0], data[:, 2], kind[:, 1]))
    # data = np.hstack((Voltage, Width)).transpose()
    # data = pd.DataFrame(data).rename(columns={0: "Input Value", 1: "Data Value", 2: "Kind"})
    # data = pd.DataFrame(data).rename(columns={0: "Input Value", 1: "Write Voltage", 2: "Pulse Width"})
    # print(data)
    # data = sns.load_dataset("planets")
    # print(data)
    # plt.figure()
    # scatter_line(data)
    # fig = sns.kdeplot(data=data, x="Value", hue="Pulse Width", multiple="fill")
    # fig.set_xlim(0, 60)
    # plt.savefig("Possibility.pdf")
    # density(data)
    # DNL(data)
    VoltageCurrent(data, "one_step_write_noise.csv", "two_step_noise.csv")
    # step 1: Total Summarized Noise = 6.12034e-06 V^2
    # step 2: Total Summarized Noise = 1.3948e-05 V^2
    # VoltageTimeVoltage(data, "one_step_noise.csv", "one_step_read_noise.csv")
    # DrawPicture("data_fig")
    # WeightMap()
    # AppliDraw()
    # CellDraw()

    return


if __name__ == '__main__':
    main()
    # bch = galois.BCH(7, 4)
    # print(bch.G)
    # print(bch.generator_poly)
    # GF = galois.GF(2 ** 3)
    # print(GF.primitive_elements)
    # print(GF.repr_table())
    # for i in [2, 4, 3, 6, 7, 5]:
    #     print(GF(i).minimal_poly())
    # fpath = Path(mpl.get_data_path(), '/Users/xxxx/Desktop/Workplace/Code/arial/arial.ttf')
    # plt.bar(["Analog Memory\nWith Cap", "Traditional CIM", "Ours"], [5.05 / 27.66, 5.05 / 2.99, 5.05])
    # plt.semilogy()
    # plt.yticks(fontproperties='Arial')
    # plt.xticks(fontproperties='Arial')
    # # plt.savefig("Buffer.pdf")
    # print([5.05 / 27.66, 5.05 / 2.99, 5.05])
    # plt.show()
