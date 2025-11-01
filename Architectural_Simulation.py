import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
import matplotlib as mpl
RRAM_AREA = 53
SRAM_AREA = 120
CAP_AREA = 1658
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42


def BufferComp(PCompTimes, BitNum, Parallelism, DataSize, WeightSize, ColParallelism):
    BufferSize = PCompTimes * (BitNum - 1 + np.log2(Parallelism)) / ColParallelism
    TemporalMem = DataSize * BitNum
    OurSize = [(np.sum(PCompTimes) + np.sum(WeightSize)) * RRAM_AREA,
                np.sum(PCompTimes) * RRAM_AREA,
                np.sum(WeightSize) * RRAM_AREA]
    # OurSize = np.sum(PCompTimes) + np.sum(WeightSize)
    TraditionalSize = [np.sum(BufferSize) * SRAM_AREA + np.sum(TemporalMem) * SRAM_AREA + np.sum(WeightSize) * RRAM_AREA,
                       np.sum(BufferSize) * SRAM_AREA + np.sum(TemporalMem) * SRAM_AREA,
                       np.sum(WeightSize) * RRAM_AREA]
    # TraditionalSize = np.sum(BufferSize) + np.sum(TemporalMem) + np.sum(WeightSize)
    MingooSize = [np.sum(PCompTimes) * CAP_AREA + np.sum(WeightSize) * SRAM_AREA,
                  np.sum(WeightSize) * SRAM_AREA,
                  np.sum(PCompTimes) * CAP_AREA]
    # MingooSize = np.sum(PCompTimes) + np.sum(WeightSize)
    ADCFreeSize = [np.sum(PCompTimes) * SRAM_AREA + np.sum(WeightSize) * RRAM_AREA * 0.5,
                   np.sum(PCompTimes) * SRAM_AREA,
                   np.sum(WeightSize) * RRAM_AREA * 0.5]
    return OurSize, TraditionalSize, MingooSize, ADCFreeSize


def Eyeriss(InputChannel, OutputChannel, WeightBit, InputBit, Kernel, FeatureMap):
    PartialSum = WeightBit + InputBit + np.ceil(np.log2(InputChannel * Kernel))
    PE_BUF = (Kernel * InputBit + InputChannel * WeightBit * Kernel + PartialSum) * OutputChannel
    InputBuf = FeatureMap * InputBit
    OutputBuf = PE_BUF * 28
    return [(PE_BUF + InputBuf + OutputBuf) * SRAM_AREA,
            (InputBuf + OutputBuf) * SRAM_AREA,
            PE_BUF * SRAM_AREA]
    # return PE_BUF + InputBuf + OutputBuf


def Systolic(InputChannel, OutputChannel, WeightBit, InputBit):
    ParallelismX = min(96, InputChannel)
    ParallelismY = min(96, OutputChannel)
    PartialSum = WeightBit + InputBit + np.ceil(np.log2(ParallelismX))
    PE_BUF = WeightBit + InputBit + PartialSum
    InputBuf = InputChannel * InputBit + InputChannel * OutputChannel * WeightBit
    OutputBuf = (PartialSum + np.ceil(np.log2(ParallelismY))) + (PartialSum + np.ceil(np.log2(InputChannel)))
    return [(PE_BUF * ParallelismX * ParallelismY + InputBuf + OutputBuf) * SRAM_AREA,
            (InputBuf + OutputBuf) * SRAM_AREA,
            (PE_BUF * ParallelismX * ParallelismY) * SRAM_AREA]
    # return PE_BUF * ParallelismX * ParallelismY + InputBuf + OutputBuf


# def Systolic(InputChannel, OutputChannel, WeightBit, InputBit):  # no hardware reuse(adder tree)
#     ParallelismX = min(96, InputChannel)
#     ParallelismY = min(96, OutputChannel)
#     PartialSum =  WeightBit + InputBit + np.ceil(np.log2(ParallelismX))
#     PE_BUF = WeightBit + InputBit + PartialSum
#     InputBuf = InputChannel * InputBit + InputChannel * OutputChannel * WeightBit
#     OutputBuf = (PartialSum + np.ceil(np.log2(ParallelismY))) * np.ceil(OutputChannel / ParallelismY) + (PartialSum + np.ceil(np.log2(InputChannel)))
#     return PE_BUF * ParallelismX * ParallelismY + InputBuf + OutputBuf


def survey(results, category_names):
    """
    Parameters
    ----------
    results : dict
        A mapping from question labels to a list of answers per category.
        It is assumed all lists contain the same number of entries and that
        it matches the length of *category_names*.
    category_names : list of str
        The category labels.
    """
    labels = list(results.keys())
    data = np.array(list(results.values()))
    data_cum = data.cumsum(axis=1)
    category_colors = plt.colormaps['RdYlGn'](
        np.linspace(0.15, 0.85, data.shape[1]))

    fig, ax = plt.subplots(figsize=(9.2, 5))
    ax.invert_yaxis()
    ax.xaxis.set_visible(True)
    # ax.semilogx()
    ax.set_xlim(0, np.sum(data, axis=1).max())

    for i, (colname, color) in enumerate(zip(category_names, category_colors)):
        widths = data[:, i]
        starts = data_cum[:, i] - widths
        rects = ax.barh(labels, widths, left=starts, height=0.5,
                        label=colname, color=color)
        r, g, b, _ = color
        text_color = 'white' if r * g * b < 0.5 else 'darkgrey'
        # ax.bar_label(rects, label_type='center', color=text_color)
    # ax.legend(ncols=len(category_names), bbox_to_anchor=(0, 1),
    #           loc='lower left', fontsize='small')
    ax.legend(bbox_to_anchor=(0, 1),
              loc='lower left', fontsize='small')
    return fig, ax


def MLP_1DCNN():
    DataSize = [16, 32, 200, 400, 784]
    WeightSize = []
    Parallelism = 4
    ColParallelism = 4
    BitNum = 4
    for i in range(len(DataSize) - 1):
        WeightSize.append(DataSize[i] * DataSize[i + 1])
    WeightSize = np.array(WeightSize)
    DataSize = np.array(DataSize)
    PCompTimes = WeightSize / Parallelism
    MLPOurSize, MLPTraditionalSize, MLPMingoo, MLPADCFree = BufferComp(PCompTimes, BitNum, Parallelism, DataSize, WeightSize, ColParallelism)
    SRAM_TOTAL = 0
    SRAM_Weight = 0
    SRAM_Temporal = 0
    for i in range(len(DataSize) - 1):
        Temp = Systolic(DataSize[i], DataSize[i + 1], 2, 4)
        SRAM_TOTAL += Temp[0]
        SRAM_Temporal += Temp[1]
        SRAM_Weight += Temp[2]
    MLPSRAM = [SRAM_TOTAL, SRAM_Temporal, SRAM_Weight]
    DataSize = [18 * 32, 4 * 28, 4, 1]
    DataSize = np.array(DataSize)
    WeightSize = [18 * 4 * 7, 4]
    WeightSize = np.array(WeightSize) * 4
    PCompTimes = WeightSize / Parallelism * 28
    CNNOurSize, CNNTraditionalSize, CNNMingoo, CNNADCFree = BufferComp(PCompTimes, BitNum, Parallelism, DataSize, WeightSize, ColParallelism)
    CNNSRAM = Eyeriss(18, 4, 2, 4, 7, DataSize[0]) + WeightSize[-1] + 4 + 2 + np.ceil(np.log2(WeightSize[-1]))
    NetKind = [
        "Analog Memory\nwith Cap",
        "SRAM Accelerator",
        "Traditional CIM",
        "ADC Free",
        "Ours"
    ]

    BufferNum = {
        "VAE Decoder": [21.7 / float("{:.2e}".format(MLPMingoo[0])),
                        1.152 / float("{:.2e}".format(MLPSRAM[0])),
                        6.002 / float("{:.2e}".format(MLPTraditionalSize[0])),
                        35.23 / float("{:.2e}".format(MLPADCFree[0])),
                        62.58 / float("{:.2e}".format(MLPOurSize[0]))],
        "1D CNN": [21.7 / float("{:.2e}".format(CNNMingoo[0])),
                   8.337 / float("{:.2e}".format(CNNSRAM[0])),
                   6.002 / float("{:.2e}".format(CNNTraditionalSize[0])),
                   35.23 / float("{:.2e}".format(CNNADCFree[0])),
                   62.58 / float("{:.2e}".format(CNNOurSize[0]))]
    }
    x = np.arange(len(NetKind))  # the label locations
    width = 0.2  # the width of the bars
    multiplier = 0
    fig, ax = plt.subplots(layout='constrained')

    for attribute, measurement in BufferNum.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width)
        ax.bar_label(rects, padding=3)
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('FoM = Power Efficiency / Buffer Area')
    # ax.set_title('Penguin attributes by species')
    ax.set_xticks(x + width, NetKind)
    ax.legend(loc='upper right')
    ax.set_axisbelow(True)
    plt.grid(axis='y', which="both")
    plt.semilogy()
    plt.savefig("FoM.pdf")
    plt.show()
    MLPDict = {
        "Analog Memory\nwith Cap": MLPMingoo[1:],
        "SRAM Accelerator": MLPSRAM[1:],
        "Traditional CIM": MLPTraditionalSize[1:],
        "ADC Free": MLPADCFree[1:],
        "Ours": MLPOurSize[1:]
    }
    CNNDict = {
        "Analog Memory\nwith Cap": CNNMingoo[1:],
        "SRAM Accelerator": CNNSRAM[1:],
        "Traditional CIM": CNNTraditionalSize[1:],
        "ADC Free": CNNADCFree[1:],
        "Ours": CNNOurSize[1:]
    }
    # survey(results=MLPDict, category_names=["Temporal Memory", "Computing Memory"])
    # plt.savefig("MLP_SpaceRatio.pdf")
    # plt.show()
    # survey(results=CNNDict, category_names=["Temporal Memory", "Computing Memory"])
    # plt.savefig("CNN_SpaceRatio.pdf")
    # plt.show()
    return


def VAE():
    InputPic = torch.randn((1, 3, 64, 64))
    InputChannel = 3
    Parallelism = 4
    ColParallelism = 4
    BitNum = 4
    latentdim = 20
    ChannelSize = [32, 64, 128, 256]
    EncoderKernelSize = 4
    EncoderStride = 2
    EncoderWeightSize = []
    DecoderKernelSize = 3
    DecoderStride = 1
    DecoderWeightSize = []
    for channel in ChannelSize:
        EncoderWeightSize.append(channel * InputChannel * (EncoderKernelSize ** 2))
        DecoderWeightSize.append(channel * InputChannel * (DecoderKernelSize ** 2))
        InputChannel = channel
    EncoderWeightSize.append(256 * latentdim * 2)
    EncoderWeightSize = np.array(EncoderWeightSize)
    DecoderWeightSize.append(4096 * latentdim)
    DecoderWeightSize.reverse()
    DecoderWeightSize = np.array(DecoderWeightSize)
    EncoderCompTimes = []
    EncoderDataSize = []
    EncoderSystolicChannel = []
    EncoderFeature = []
    DecoderCompTimes = []
    DecoderDataSize = []
    DecoderSystolicChannel = []
    DecoderFeature = []
    InputChannel = InputPic.size(1)
    EncoderFeature.append(InputPic.size(2) ** 2)
    for channel in ChannelSize:
        TempInput = nn.functional.unfold(InputPic, EncoderKernelSize, stride=EncoderStride).transpose(1, 2)
        TempInput = TempInput.reshape((-1, TempInput.size(2)))
        EncoderSystolicChannel.append(TempInput.size(1))
        EncoderCompTimes.append(TempInput.size(0) * TempInput.size(1) / Parallelism)
        InputPic = nn.Conv2d(InputChannel, channel, EncoderKernelSize, EncoderStride)(InputPic)
        EncoderDataSize.append(InputPic.size(0) * InputPic.size(1) * InputPic.size(2) * InputPic.size(3))
        EncoderFeature.append(InputPic.size(2) ** 2)
        InputChannel = channel
    EncoderCompTimes.append(EncoderWeightSize[-1] / Parallelism)
    EncoderDataSize.append(256)
    EncoderDataSize.append(latentdim)
    EncoderCompTimes = np.array(EncoderCompTimes)
    DecoderDataSize.append(4096)
    DecoderCompTimes.append(DecoderWeightSize[0] / Parallelism)
    ChannelSize.reverse()
    InputPic = torch.randn((1, 256, 4, 4))
    DecoderFeature.append(InputPic.size(2) ** 2)
    InputChannel = 256
    for channel in ChannelSize:
        InputPic = nn.UpsamplingNearest2d(scale_factor=2)(InputPic)
        TempInput = nn.functional.unfold(InputPic, DecoderKernelSize, stride=DecoderStride).transpose(1, 2)
        TempInput = TempInput.reshape((-1, TempInput.size(2)))
        DecoderSystolicChannel.append(TempInput.size(1))
        DecoderCompTimes.append(TempInput.size(0) * TempInput.size(1) / Parallelism)
        InputPic = nn.Conv2d(InputChannel, channel, DecoderKernelSize, DecoderStride, padding="same")(InputPic)
        DecoderDataSize.append(InputPic.size(0) * InputPic.size(1) * InputPic.size(2) * InputPic.size(3))
        DecoderFeature.append(InputPic.size(2) ** 2)
        InputChannel = channel
    DecoderCompTimes = np.array(DecoderCompTimes)
    EncoderOurSize, EncoderTraditionalSize, EncoderMingoo, EncoderADCFree = \
        BufferComp(EncoderCompTimes, BitNum, Parallelism, EncoderDataSize, EncoderWeightSize, ColParallelism)
    DecoderOurSize, DecoderTraditionalSize, DecoderMingoo, DecoderADCFree = \
        BufferComp(DecoderCompTimes, BitNum, Parallelism, DecoderDataSize, DecoderWeightSize, ColParallelism)
    VAEOurSize = [EncoderOurSize[i] + DecoderOurSize[i] for i in range(3)]
    VAETraditionalSize = [EncoderTraditionalSize[i] + DecoderTraditionalSize[i] for i in range(3)]
    VAEMingoo = [EncoderMingoo[i] + DecoderMingoo[i] for i in range(3)]
    VAEADCFree = [EncoderADCFree[i] + DecoderADCFree[i] for i in range(3)]
    EncoderSystolicTotal = 0
    EncoderSystolicTemporal = 0
    EncoderSystolicWeight = 0
    EncoderEyerissTotal = 0
    EncoderEyerissTemporal = 0
    EncoderEyerissWeight = 0
    ChannelSize.reverse()
    for i in range(len(EncoderSystolicChannel) - 1):
        Temp = Systolic(EncoderSystolicChannel[i], EncoderSystolicChannel[i + 1], 2, 4)
        EncoderSystolicTotal += Temp[0]
        EncoderSystolicTemporal += Temp[1]
        EncoderSystolicWeight += Temp[2]
        Temp = Eyeriss(ChannelSize[i], ChannelSize[i + 1], 2, 4, 4 ** 2, EncoderFeature[i])
        EncoderEyerissTotal += Temp[0]
        EncoderEyerissTemporal += Temp[1]
        EncoderEyerissWeight += Temp[2]
    Temp = Systolic(256, latentdim, 2, 4)
    EncoderSystolicTotal += Temp[0] * 2
    EncoderSystolicTemporal += Temp[1] * 2
    EncoderSystolicWeight += Temp[2] * 2
    EncoderEyerissTotal += Temp[0] * 2
    EncoderEyerissTemporal += Temp[1] * 2
    EncoderEyerissWeight += Temp[2] * 2
    DecoderSystolicTotal = 0
    DecoderSystolicTemporal = 0
    DecoderSystolicWeight = 0
    DecoderEyerissTotal = 0
    DecoderEyerissTemporal = 0
    DecoderEyerissWeight = 0
    Temp = Systolic(latentdim, 4096, 2, 4)
    DecoderSystolicTotal += Temp[0]
    DecoderSystolicTemporal += Temp[1]
    DecoderSystolicWeight += Temp[2]
    DecoderEyerissTotal += Temp[0]
    DecoderEyerissTemporal += Temp[1]
    DecoderEyerissWeight += Temp[2]
    ChannelSize.reverse()
    for i in range(len(DecoderSystolicChannel) - 1):
        Temp = Systolic(DecoderSystolicChannel[i], EncoderSystolicChannel[i + 1], 2, 4)
        DecoderSystolicTotal += Temp[0]
        DecoderSystolicTemporal += Temp[1]
        DecoderSystolicWeight += Temp[2]
        Temp = Eyeriss(ChannelSize[i], ChannelSize[i + 1], 2, 4, 3 ** 2, DecoderFeature[i])
        DecoderEyerissTotal += Temp[0]
        DecoderEyerissTemporal += Temp[1]
        DecoderEyerissWeight += Temp[2]
    VAESystolic = [EncoderSystolicTotal + DecoderSystolicTotal,
                   EncoderSystolicTemporal + DecoderSystolicTemporal,
                   EncoderSystolicWeight + DecoderSystolicWeight]
    VAEEyeriss = [EncoderEyerissTotal + DecoderEyerissTotal,
                  EncoderEyerissTemporal + DecoderEyerissTemporal,
                  EncoderEyerissWeight + DecoderEyerissWeight]
    NetKind = [
        "SRAM Convolution\nAccelerator",
        "SRAM Systolic\nAccelerator",
        "Analog Memory\nwith Cap",
        "Traditional CIM",
        "ADC Free",
        "Ours"
    ]

    BufferNum = {
        "VAE Decoder": [8.337 / float("{:.2e}".format(VAEEyeriss[0])),
                        1.152 / float("{:.2e}".format(VAESystolic[0])),
                        31.04 / float("{:.2e}".format(VAEMingoo[0])),
                        6.002 / float("{:.2e}".format(VAETraditionalSize[0])),
                        35.23 / float("{:.2e}".format(VAEADCFree[0])),
                        62.58 / float("{:.2e}".format(VAEOurSize[0]))]
    }
    # Ours, _, _, _ = BufferComp(256, BitNum, Parallelism, 32, 32 * 4, 4)
    # print(256 / (16e-9 * 8) / (Ours[1] * (28 * 28)))
    # x = np.arange(len(NetKind))  # the label locations
    # width = 0.2  # the width of the bars
    # multiplier = 0
    # fig, ax = plt.subplots(layout='constrained')

    # for attribute, measurement in BufferNum.items():
    #     offset = width * multiplier
    #     rects = ax.bar(x + offset, measurement, width)
    #     ax.bar_label(rects, padding=3)
    #     multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    # ax.set_ylabel('FoM = Power Efficiency / Buffer Area')
    # ax.set_xticks(x + width, NetKind)
    # ax.legend(loc='upper right')
    # ax.set_axisbelow(True)
    # plt.grid(axis='y', which="both")
    # plt.semilogy()
    # plt.savefig("FoM.pdf")
    # plt.show()
    VAEDict = {
        "SRAM Convolution\nAccelerator":[VAEEyeriss[1], VAEEyeriss[2]],
        "SRAM Systolic\nAccelerator": VAESystolic[1:],
        "Analog Memory\nwith Cap": VAEMingoo[1:],
        "Traditional CIM": VAETraditionalSize[1:],
        "ADC Free": VAEADCFree[1:],
        "Ours": VAEOurSize[1:]
    }
    # print(VAEADCFree[0] / VAEOurSize[0])
    # print(VAEDict)
    # survey(results=VAEDict, category_names=["Temporal Memory", "Computing Memory"])
    # plt.savefig("VAE_SpaceRatio.pdf")
    # plt.show()
    return


def main():
    # MLP_1DCNN()
    VAE()
    return


if __name__ == '__main__':
    main()