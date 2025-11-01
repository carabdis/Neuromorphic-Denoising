import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


def AnalogNoise(Input_dist, Analog_dist):
    avg_dist, sigma_dist = Analog_dist
    SamplingGroup = np.round(Input_dist / 25.2 * 60)
    OutputSampling = []
    SampleNum = 1000
    for item in SamplingGroup:
        OutputSampling.append(np.clip(np.random.normal(avg_dist[int(item)], sigma_dist[int(item)], SampleNum), 0, 25.2))
    OutputSampling = np.hstack(OutputSampling)
    OutputSampling = OutputSampling[np.random.randint(0, len(OutputSampling), SampleNum * 10)]
    return OutputSampling


def QuantizationNoise(Input_dist, Quant_dist):
    Output_dist = [0] * Input_dist.shape[0]
    SampleNum = 1000
    SampleValue = []
    for _ in range(SampleNum * 4):
        rand = np.random.rand()
        poss = 0
        for j in range(4):
            if Input_dist[j] + poss > rand:
                SampleValue.append(j)
                break
            else:
                poss += Input_dist[j]
    CoumputeValue = []
    for i in range(SampleNum):
        CoumputeValue.append(np.clip(int(np.sum(SampleValue[i * 4:(i + 1) * 4]) / 4), 0, 3))
    Output_dist = np.sum(Quant_dist[CoumputeValue], axis=0) / SampleNum
    return Output_dist


def NoiseAccumulate():
    data = pd.read_csv("Data.csv")
    AnalogUpperLim = 4
    AnalogLowerLim = 0.1
    RefValue = np.log10(AnalogUpperLim / AnalogLowerLim)
    AnalogPulse = []
    for index, row in data.iterrows():
        AnalogPulse.append(np.clip(np.log10(row["Read Current"] / AnalogLowerLim),
                                   0, RefValue) / RefValue * 25.2)
    data.insert(4, "Analog Pulse", AnalogPulse, True)

    NumCount = [0] * 61
    RawDist = [[] for i in range(61)]
    Average = [0] * 61
    Sigma = [0] * 61
    InitialProb = np.zeros((61, 4))
    TransMat = np.zeros((4, 4))
    PulseDict = {round(8.4 * i, 3): (3 - i) for i in range(4)}
    InputSampling = []
    for index, row in data.iterrows():
        NumCount[int(row['Value'])] += 1
        RawDist[int(row['Value'])].append(row["Analog Pulse"])
        PulseKind = PulseDict[row['Pulse Width']]
        InitialProb[int(row['Value'])][PulseKind] += 1
    EmptyPlace = []
    Min = 61
    Max = 0
    for i in range(61):
        if np.sum(InitialProb[i]) != 0:
            InitialProb[i] /= np.sum(InitialProb[i])
            Average[i] = np.mean(RawDist[i])
            Sigma[i] = np.std(RawDist[i])
            Min = min(i, Min)
            Max = i
        else:
            EmptyPlace.append(i)
    for item in EmptyPlace:
        if item < Min:
            InitialProb[item] = InitialProb[Min]
            Average[item] = Average[Min] + (Min - item) * (Average[Min] - Average[Min + 1])
            Sigma[item] = Sigma[Min]
        elif item > Max:
            InitialProb[item] = InitialProb[Max]
            Average[item] = Average[Max] - (item - Max) * (Average[Min - 1] - Average[Min])
            Sigma[item] = Sigma[Max]
        else:
            InitialProb[item] = (InitialProb[item - 1] + InitialProb[item + 1]) / 2
    print(Sigma[30] / Average[30])
    for i in range(InitialProb.shape[0]):
        print(str(i) + '\t[', end='')
        for j in range(InitialProb.shape[1]):
            print(InitialProb[i][j], end=', ')
        print('],')
    for i in range(4):
        TransMat[i] = InitialProb[15 * i]
        for j in range(15):
            if InitialProb[15 * i + j][i] > TransMat[i][i]:
                TransMat[i] = InitialProb[15 * i + j]
    SampleNum = 1000
    for i in range(61):
        InputSampling.append(np.clip(np.random.normal(Average[i], Sigma[i], SampleNum * 10), 0, 25.2))
    AnalogResult = []
    QuantResult = []
    Target = 40
    IterationTimes = 10
    for i in range(IterationTimes):
        if i == 0:
            Analog = AnalogNoise(InputSampling[Target], [Average, Sigma])
            Quant = QuantizationNoise(InitialProb[Target], TransMat)
        else:
            Analog = AnalogNoise(AnalogResult[i - 1], [Average, Sigma])
            Quant = QuantizationNoise(QuantResult[i - 1], TransMat)
        AnalogResult.append(Analog)
        QuantResult.append(Quant)
    AnalogData = (25.2 - AnalogResult[-1]) / 8.4 - Target / 15
    return
    plt.hist(AnalogData, weights=np.zeros_like(AnalogData) + 1. / AnalogData.size)
    plt.savefig("Analog_Error.pdf")
    plt.show()
    plt.bar([i - int(Target / 15) for i in range(61)], QuantResult[-1])
    plt.savefig("Quant_Error.pdf")
    plt.show()
    return


def Encoding(Info, TimeStep=7000, InfoMax = 60):
    SampleNum = len(Info)
    PAM = np.zeros((SampleNum, TimeStep))
    PWM = np.zeros((SampleNum, TimeStep))
    SPIKE = np.zeros((SampleNum, TimeStep))
    PAMIndent = 2000
    PWMIndent = 100
    SpikeIndent = 50
    PreZero = 600
    PAM[:, PreZero:PreZero + PAMIndent] = np.ones((SampleNum, PAMIndent)) * Info.reshape(SampleNum, 1) / InfoMax
    for i in range(len(Info)):
        PWM[i, PWMIndent:PWMIndent * (Info[i] + 1)] = np.ones(PWMIndent * Info[i])
        for j in range(Info[i]):
            SPIKE[i, SpikeIndent * 2 * (j + 1):SpikeIndent * 2 * (j + 1) + SpikeIndent] = np.ones(int(SpikeIndent))
    return PAM, PWM, SPIKE


def Storage(Info, InfoMax = 60, Threshold = 0.8):
    PAM, PWM, SPIKE = Info
    SampleNum, TimeStep = PAM.shape
    PAMIndent = 2000
    PWMIndent = 100
    SpikeIndent = 50
    PreZero = 600
    Index = np.random.randint(PAMIndent) + PreZero
    DataPAM = PAM[:, Index]
    IndexPWM = (PWM > Threshold)
    DataPWM = np.sum(IndexPWM, axis=1) / PWMIndent / InfoMax
    IndexSPIKE = (SPIKE > Threshold)
    DataSPIKE = np.sum(IndexSPIKE, axis=1) / SpikeIndent / InfoMax
    return DataPAM, DataPWM, DataSPIKE


def Read(Info, Ratio=0.1, InfoMax = 60, AccTime = 1000):
    PAM, PWM, SPIKE = Info
    SampleNum = len(PAM)
    AccNoise = np.random.randn(SampleNum * AccTime).reshape((SampleNum, AccTime)) * Ratio
    AccPWM = np.clip(np.tile(PWM, (AccTime, 1)).transpose() + AccNoise, 0, None)
    AccSPIKE = np.clip(np.tile(SPIKE, (AccTime, 1)).transpose() + AccNoise, 0, None)
    OutPAM = np.clip(PAM + np.random.randn(SampleNum) * Ratio, 0, None) * InfoMax
    Step = 1 / AccTime
    SampleStep = 10
    SpikeThreshold = 16
    OutPWM = np.clip(PWM + np.random.randn(SampleNum) * Ratio / np.sqrt(2), 0, None) * InfoMax
    OutQuantPWM = np.zeros(PWM.shape)
    OutSPIKE = np.zeros(SPIKE.shape)
    for i in range(SampleNum):
        count_PWM = 0
        count_SPIKE = 0
        cap_PWM = 0
        cap_SPIKE = 0
        for j in range(AccTime):
            if j % SampleStep == SampleStep - 1:
                if cap_PWM >= AccPWM[i, j]:
                    OutQuantPWM[i] = count_PWM * InfoMax * Step * SampleStep
                else:
                    count_PWM += 1
            cap_PWM += Step * (1 + AccNoise[i, j])
            if cap_SPIKE > SpikeThreshold:
                cap_SPIKE -= SpikeThreshold
                count_SPIKE += 1
            else:
                cap_SPIKE += AccSPIKE[i, j]
        OutSPIKE[i] = count_SPIKE / AccTime * SpikeThreshold * InfoMax
    return OutPAM, OutPWM, OutQuantPWM, OutSPIKE


def main():
    # NoiseAccumulate()
    SNRList = [[], [], [], []]
    grid = 100
    Limit = 0.5
    for step in range(1, grid + 1):
        print(step, '/', grid)
        Info = np.random.randint(0, 60, 1000)
        PAM, PWM, SPIKE = Encoding(Info)
        SampleNum, TimeStep = PAM.shape
        ratio = Limit / grid * step
        # ratio = 0
        TransNoise = np.random.randn(SampleNum * TimeStep).reshape(PAM.shape) * ratio
        NoisyPAM = np.clip(TransNoise + PAM, 0, None)
        NoisyPWM = np.clip(TransNoise + PWM, 0, None)
        NoisySPIKE = np.clip(TransNoise + SPIKE, 0, None)
        StoPAM, StoPWM, StoSPIKE = Storage([NoisyPAM, NoisyPWM, NoisySPIKE])
        FinPAM, FinPWM, FinQuantPWM, FinSPIKE = Read((StoPAM, StoPWM, StoSPIKE), Ratio=ratio)
        ErrorPAM = abs(FinPAM - Info)
        ErrorPWM = abs(FinPWM- Info)
        ErrorQuantPWM = abs(FinQuantPWM - Info)
        ErrorSPIKE = abs(FinSPIKE - Info)
        SNRPAM = np.log10(np.mean(Info) / np.mean(ErrorPAM)) * 20
        SNRPWM = np.log10(np.mean(Info) / np.mean(ErrorPWM)) * 20
        SNRQuantPWM = np.log10(np.mean(Info) / np.mean(ErrorQuantPWM)) * 20
        SNRSPIKE = np.log10(np.mean(Info) / np.mean(ErrorSPIKE)) * 20
        SNRList[0].append(SNRPAM)
        SNRList[1].append(SNRPWM)
        SNRList[2].append(SNRQuantPWM)
        SNRList[3].append(SNRSPIKE)
    x = Limit / grid * np.arange(1, grid + 1)
    SNR = np.array(SNRList).transpose()
    # s = np.array([SNRList[0], SNRList[1], SNRList[3]]).transpose()
    data = pd.DataFrame(SNR, x, columns=["PAM", "PWM", "Quant PWM", "Spike"])
    # data = pd.DataFrame(s, x, columns=["PAM", "PWM", "Spike"])
    sns.lineplot(data=data, palette="tab10", linewidth=2.5)
    plt.axhline(15.201691738550627)
    data.to_csv("SNR.csv")
    plt.savefig("SNR.pdf")
    plt.show()
    return


if __name__ == '__main__':
    main()
