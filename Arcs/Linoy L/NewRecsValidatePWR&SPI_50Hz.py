import numpy
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import numpy as np
import heapq
import pandas as pd
import csv
import statistics
import sklearn
from sklearn.metrics import classification_report, confusion_matrix
from numpy.random import randn
from numpy.random import seed
from scipy.stats import pearsonr
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from sklearn import preprocessing
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix , f1_score , precision_score , recall_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import export_graphviz
import pydot
import SPI_Reading
import math
from scipy import signal
import cmath
#import noam


def first_gt_index(lst, k):
    # creating a heap from the list
    heap = list(lst)
    heapq.heapify(heap)
    # using heappop() to find index of first element
    # just greater than k
    # print(heap)
    for i, val in enumerate(lst):
        if val >= k:
            res = i
            break
    else:
        res = None

    return res

def first_gt_index1(lst, k):
    # creating a heap from the list
    heap = list(lst)
    heapq.heapify(heap)
    # using heappop() to find index of first element
    # just greater than k
    # print(heap)
    for i, val in enumerate(lst):
        if val > k:
            res = i
            break
    else:
        res = None

    return res

def Counter_PAC_RMS_calcNew(IAC_RMS,timeRmsIac,buffer_size = 50,windowSize = 7, filter_size=28,TH_Iac = 50):
    IacCounterVector = [0 for x in range(buffer_size - windowSize - filter_size )]
    timeCounterVec = []
    RaiseEnergyINDX = len(IAC_RMS)
    k = 0
    for index in range(buffer_size, len(IAC_RMS)):
        IacCounter = 0
        ind = index - windowSize - filter_size
        timeCounter = timeRmsIac[index]
        timeCounterVec.append(timeCounter)
        # IAC_RMS Fill buff
        bufferI = IAC_RMS[ind:ind + buffer_size]
        windowI = bufferI[-windowSize:]
        filterI = bufferI[-(windowSize+filter_size): -windowSize]


        AVG_in_filter_I = np.min(filterI)  #np.mean(filterI) #(np.min(filterI) + np.mean(filterI))/2 #+2*np.std(filterI)#np.mean(filterI)

        for value in windowI:
            if value < (AVG_in_filter_I- TH_Iac):
                IacCounter+=1
        IacCounterVector +=[IacCounter]

        # if IacCounterVector >= 11 and k == 0 :
        #     RaiseEnergyINDX = index
        #     k = k+1

    return IacCounterVector,timeCounterVec

def PAC_RMS_Drop_calcNew(IAC_RMS,timeS, windowSize = 7, filter_size=28, samTH = 5):
    buffer_size = 50 # 50
    min_in_filterEachbuffer = []
    VecThresh = [0 for x in range(buffer_size - windowSize - filter_size )]
    timeWindVec = []
    for index in range(buffer_size, len(IAC_RMS)):
        timeWind = timeS[index]
        timeWindVec.append(timeWind)
        ind = index - windowSize - filter_size
        buffer1 = IAC_RMS[ind:ind + buffer_size]
        window1 = buffer1[-windowSize:]
        filter1 = buffer1[-(windowSize+filter_size): -windowSize]
        min_in_filter =  np.min(filter1) #np.mean(filter1) #(np.min(filter1) + np.mean(filter1))/2 #np.min(filter1) #+2*np.std(filter1) #np.min(filter1) #np.mean(filter1) #np.mean(filter1)
        min_in_filterEachbuffer += [min_in_filter]
        SortWind = numpy.sort(window1)
        SortWindFlip = SortWind
        if len(SortWindFlip) > samTH:
            K_sampValWin = SortWindFlip[samTH-1]
            VecThresh += [K_sampValWin - min_in_filter]
        if (len(SortWindFlip) <= samTH) & (SortWindFlip.size > 0):
            K_sampValWin = SortWindFlip[0]
            VecThresh += [K_sampValWin - min_in_filter]
        index += 1
    VecThresh = [0 if x > 0 else float(x) for x in VecThresh]
    return np.abs(VecThresh),timeWindVec

def Counter_IAC_RMS_calc(IAC_RMS,timeRmsIac,buffer_size = 50,windowSize = 20, filter_size=15,TH_Iac = 0.2):
    IacCounterVector = []
    timeCounterVec = []
    for index in range(buffer_size, len(IAC_RMS)):
        IacCounter = 0
        ind = index - buffer_size
        timeCounter = timeRmsIac[index]
        timeCounterVec.append(timeCounter)
        # IAC_RMS Fill buff
        bufferI = IAC_RMS[ind:ind + buffer_size]
        windowI = bufferI[-windowSize:]
        filterI = bufferI[-(windowSize+filter_size): -windowSize]


        AVG_in_filter_I = np.mean(filterI)

        for value in windowI:
            if value < (AVG_in_filter_I- TH_Iac):
                IacCounter+=1
        IacCounterVector +=[IacCounter]

    return IacCounterVector,timeCounterVec

def Counter_IAC_RMS_calcNew(IAC_RMS,timeRmsIac,buffer_size = 50,windowSize = 7, filter_size=28,TH_Iac = 0.4):
    IacCounterVector = [0 for x in range(buffer_size - windowSize - filter_size )]
    timeCounterVec = []
    RaiseEnergyINDX = len(IAC_RMS)
    k = 0
    for index in range(buffer_size, len(IAC_RMS)):
        IacCounter = 0
        ind = index - windowSize - filter_size
        timeCounter = timeRmsIac[index]
        timeCounterVec.append(timeCounter)
        # IAC_RMS Fill buff
        bufferI = IAC_RMS[ind:ind + buffer_size]
        windowI = bufferI[-windowSize:]
        filterI = bufferI[-(windowSize+filter_size): -windowSize]


        AVG_in_filter_I = np.mean(filterI) #+2*np.std(filterI)#np.mean(filterI)

        for value in windowI:
            if value < (AVG_in_filter_I- TH_Iac):
                IacCounter+=1
        IacCounterVector +=[IacCounter]

        # if IacCounterVector >= 11 and k == 0 :
        #     RaiseEnergyINDX = index
        #     k = k+1

    return IacCounterVector,timeCounterVec

def IAC_RMS_Drop_calcNew(IAC_RMS,timeS, windowSize = 7, filter_size=28, samTH = 5):
    buffer_size = 50 # 50
    min_in_filterEachbuffer = []
    VecThresh = [0 for x in range(buffer_size - windowSize - filter_size )]
    timeWindVec = []
    for index in range(buffer_size, len(IAC_RMS)):
        timeWind = timeS[index]
        timeWindVec.append(timeWind)
        ind = index - windowSize - filter_size
        buffer1 = IAC_RMS[ind:ind + buffer_size]
        window1 = buffer1[-windowSize:]
        filter1 = buffer1[-(windowSize+filter_size): -windowSize]
        min_in_filter =  np.min(filter1) #+2*np.std(filter1) #np.min(filter1) #np.mean(filter1) #np.mean(filter1)
        min_in_filterEachbuffer += [min_in_filter]
        SortWind = numpy.sort(window1)
        SortWindFlip = SortWind
        if len(SortWindFlip) > samTH:
            K_sampValWin = SortWindFlip[samTH-1]
            VecThresh += [K_sampValWin - min_in_filter]
        if (len(SortWindFlip) <= samTH) & (SortWindFlip.size > 0):
            K_sampValWin = SortWindFlip[0]
            VecThresh += [K_sampValWin - min_in_filter]
        index += 1
    VecThresh = [0 if x > 0 else float(x) for x in VecThresh]
    return np.abs(VecThresh),timeWindVec

def Counter_VDC_IAC_Energy_calcOldMethod(log_energy,IAC_RMS,VDC_Slow,timeEnergy,buffer_size = 50,windowSize = 20, filter_size=15, TH_Energy = 12,TH_Iac = 0.2,TH_Vdc = 0.005):
    EnergyCounterVector = []
    IacCounterVector = []
    VdcCounterVector = []
    timeCounterVec = []
    for index in range(buffer_size, len(log_energy)):
        EnergyCounter = 0
        IacCounter = 0
        VdcCounter = 0
        ind = index - buffer_size
        timeCounter = timeEnergy[index]
        timeCounterVec.append(timeCounter)
        # Energy Fill buff
        bufferE = log_energy[ind:ind + buffer_size]
        windowE = bufferE[-windowSize:]
        filterE = bufferE[-(windowSize+filter_size): -windowSize]
        # IAC_RMS Fill buff
        bufferI = IAC_RMS[ind:ind + buffer_size]
        windowI = bufferI[-windowSize:]
        filterI = bufferI[-(windowSize+filter_size): -windowSize]
        # VDC_Slow Fill buff
        bufferV = VDC_Slow[ind:ind + buffer_size]
        windowV = bufferV[-windowSize:]
        filterV = bufferV[-(windowSize+filter_size): -windowSize]

        min_filterEnergy = np.min(filterE)
        AVG_in_filter_I = np.mean(filterI)
        AVG_in_filter_V = np.mean(filterV)
        for value in windowE:
            if value >= (min_filterEnergy+ TH_Energy):
                EnergyCounter+=1
        EnergyCounterVector +=[EnergyCounter]

        for value in windowI:
            if value < (AVG_in_filter_I- TH_Iac):
                IacCounter+=1
        IacCounterVector +=[IacCounter]

        for value in windowV:
            if value < (AVG_in_filter_V- TH_Vdc):
                VdcCounter+=1
        VdcCounterVector +=[VdcCounter]

    return EnergyCounterVector,IacCounterVector,VdcCounterVector,timeCounterVec



def initialize_fig(row=4, col=1, plots_per_pane=4, shared_xaxes=True, subplot_titles=['Rx', 'Rx', 'Rx', 'Rx']):
    all_specs = np.array([[{"secondary_y": True}] for x in range((row * col))])
    all_specs_reshaped = (np.reshape(all_specs, (col, row)).T).tolist()
    fig = make_subplots(rows=row, cols=col, specs=all_specs_reshaped, shared_xaxes=shared_xaxes,
                        subplot_titles=subplot_titles)

    return fig

def df_time_reset(dfx, x_col):
    """
        Function resets data chunk time vector
        `dfx`   - Pandas Data Frame
        `x_col` - exact Time col name
        Example of usage :
            df1 = df_time_reset(df1,'Time')
    """
    prop_df = dfx.copy(deep=True)
    prop_df[x_col] = prop_df[x_col] - dfx[x_col].iloc[0]
    return prop_df

def Counter_Energy_calcCorrect(log_energy,timeEnergy,buffer_size = 50,windowSize = 20, filter_size=15, TH_Energy = 12):
    EnergyCounterVector = []
    timeCounterVec = []
    for index in range(buffer_size, len(log_energy)-1):
        EnergyCounter = 0
        ind = index - buffer_size
        timeCounter = timeEnergy[index]
        timeCounterVec.append(timeCounter)
        # Energy Fill buff
        bufferE = log_energy[ind:ind + buffer_size]
        windowE = bufferE[-windowSize:]
        filterE = bufferE[-(windowSize+filter_size): -windowSize]
        min_filterEnergy = np.min(filterE)

        for value in windowE:
            if value >= (min_filterEnergy+ TH_Energy):
                EnergyCounter+=1
        EnergyCounterVector +=[EnergyCounter]
    return EnergyCounterVector,timeCounterVec

def Counter_Iac_calcCorrect(RmsIac,time_RmsIac,buffer_size = 50,windowSize = 20, filter_size=15, TH_Iac = 0.2):
    IacCounterVector = []
    timeCounterVec = []
    for index in range(buffer_size, len(RmsIac)-1):
        IacCounter = 0
        ind = index - buffer_size
        timeCounter = time_RmsIac[index]
        timeCounterVec.append(timeCounter)
        # Iac Fill buff
        bufferI = RmsIac[ind:ind + buffer_size]
        windowI = bufferI[-windowSize:]
        filterI = bufferI[-(windowSize+filter_size): -windowSize]
        AVG_in_filter_I = np.mean(filterI)

        for value in windowI:
            if value < (AVG_in_filter_I- TH_Iac):
                IacCounter+=1
        IacCounterVector +=[IacCounter]
    return IacCounterVector,timeCounterVec


def energy_rise_calcCorrect(log_energy, timeEnergy, windowSize=20, filter_size=15, samTH=12, buffer_size=50):
    VecThresh = []
    timeWindVec = []
    for index in range(buffer_size, len(log_energy)-1):
        timeWind = timeEnergy[index ]
        timeWindVec.append(timeWind)
        ind = index - buffer_size
        buffer1 = log_energy[ind:ind + buffer_size]
        filter1 = buffer1[-(windowSize + filter_size): -windowSize]
        window1 = buffer1[-windowSize:]
        min_in_filter = min(filter1)
        SortWind = numpy.sort(window1)
        SortWindFlip = np.flip(SortWind)
        if len(SortWindFlip) > samTH:
            K_sampValWin = SortWindFlip[samTH - 1]
            VecThresh += [K_sampValWin - min_in_filter]
        if (len(SortWindFlip) <= samTH) & (SortWindFlip.size > 0):
            K_sampValWin = SortWindFlip[0]
            VecThresh += [K_sampValWin - min_in_filter]
        index += 1
    VecThresh = [0 if x < 0 else float(x) for x in VecThresh]
    return VecThresh, timeWindVec

def cut_indxS(RX_RAW,Vdc_fast1,Fs_SPI):
    indicesR11 = np.where(np.array(RX_RAW) > 1000)[0].tolist()
    indicesR22 = np.where(np.array(RX_RAW) > 2000)[0].tolist()

    if len(indicesR22) > 2:
        indicesR22 = np.min([indicesR22[1] + 1000, len(RX_RAW)])  # indicesR[0] + 260000000000000000000000000
        indicesStop = indicesR22
        return indicesStop
    if len(indicesR11) != 0:
        indicesR11 = np.min([indicesR11[0] + 2200, len(RX_RAW)])  # indicesR[0] + 260000000000000000000000000
        indicesStop = indicesR11
        return indicesStop


    # indicesS_R = np.where(np.abs(np.array(RX_RAW)) > 340)[0].tolist()
    #
    # if np.max(np.array(RX_RAW)) < 1000 and len(indicesS_R)>0 : #len(indicesS_R) > 0:
    #     indicesS_R = np.min([indicesS_R[0] + 3500, len(RX_RAW)])  # indicesR[0] + 260000000000000000000000000
    #     indicesStop = indicesS_R
    #     return indicesStop

    # indicesR = np.where(np.array(np.abs(RX_RAW)) > 820)[0].tolist()
    indicesR = np.where(np.array(np.abs(RX_RAW)) > 830)[0].tolist()##############
    if len(indicesR) == 0:
        indicesR = len(RX_RAW)
    else:
        # indicesR = np.min([indicesR[0] + 1300, len(RX_RAW)])  # indicesR[0] + 260000000000000000000000000
        indicesR = np.min([indicesR[0] + 1800, len(RX_RAW)])  # indicesR[0] + 260000000000000000000000000 3500
        indicesStop = indicesR
        return indicesStop

    # indicesDiffR = np.where(np.abs(np.diff(np.array(RX_RAW))) > 205)[0].tolist()
    # indicesDiffR = np.where(np.abs(np.diff(np.array(RX_RAW))) > 700)[0].tolist()
    # if len(indicesDiffR) == 0:
    #     indicesDiffR = len(RX_RAW)
    # else:
    #     # indicesDiffR = np.min([indicesDiffR[0] + 3500, len(RX_RAW)])
    #     indicesDiffR = np.min([indicesDiffR[0] + 4500, len(RX_RAW)])
    indicesDiffR = np.where(np.abs(np.diff(np.array(RX_RAW))) > 700)[0].tolist()
    if len(indicesDiffR) > 1:
        indicesDiffR = np.min([indicesDiffR[1] + 3500, len(RX_RAW)])
    else:
        # indicesDiffR = np.min([indicesDiffR[0] + 3500, len(RX_RAW)])
        # indicesDiffR = np.min([indicesDiffR[0] + 3500, len(RX_RAW)])
        indicesDiffR = len(RX_RAW)



    indices = np.where(np.array(Vdc_fast1) > 765)[0].tolist()
    if len(indices) == 0:
        indices = len(Vdc_fast1)
    else:
        indices = indices[0] + 1000 ##########

    indices2 = np.where(np.array(Vdc_fast1) < 720)[0].tolist()
    if len(indices2) == 0:
        indices2 = len(Vdc_fast1)
    else:
        indices2 = indices2[0] + 1000 ##########

    ############
    indicesR2 = np.where(np.array(RX_RAW) > 1500)[0].tolist()  #####
    if len(indicesR2) == 0:
        indicesR2 = len(RX_RAW)
    else:
        indicesR2 = np.min([indicesR2[0] + 2000, len(RX_RAW)]) ###### was 2700
    ######
    ####
    Min_I = int(Fs_SPI * 3)
    if indicesR < Min_I:
        indicesR = len(RX_RAW)
    if indices < Min_I:
        indices = len(RX_RAW)
    if indices2 < Min_I:
        indices2 = len(RX_RAW)
    if indicesDiffR < Min_I:
        indicesDiffR = len(RX_RAW)
    if indicesR2 < Min_I:
        indicesR2 = len(RX_RAW)

    indicesStop = np.min([indicesR,indicesR2, indices, indices2, indicesDiffR])
    return indicesStop

def VdcProcess(Vdc,FsAfterDownsample):
    # current method
    # FsAfterDownsample : 16667Hz
    # Vdc : VdcFast
    arcbw = 35
    Mean_Size = int(FsAfterDownsample / arcbw)
    alpha = 1e-4
    VdcVec = [Vdc[0]]
    for index in range(1, len(Vdc)):
        VdcVec.append(alpha * Vdc[index] + (1 - alpha) * VdcVec[index - 1])
    # Fs = FsAfterDownsample
    # for index, val in enumerate(Vdc[1:]):
    #     VdcIIR = alpha * val + (1 - alpha) * LastVdc
    #     VdcVec.append(VdcIIR)
    #     LastVdc = VdcIIR
    Fs = FsAfterDownsample
    T = 1 / Fs
    N = len(VdcVec)
    time_ = np.linspace(0, N * T, N)
    VdcVecMean = []
    for index in range(0, len(VdcVec) - Mean_Size, Mean_Size):
        bufferVdc = VdcVec[index:index + Mean_Size]
        MeanVdc = np.mean(bufferVdc)
        VdcVecMean.append(MeanVdc)
    Fs = arcbw
    T = 1 / Fs
    N = len(VdcVecMean)
    time = np.linspace(0, N * T, N)

    return VdcVecMean, time , VdcVec,time_

def VDC_Drop_calc(VDC_Slow,timeS, windowSize = 20, filter_size=15, samTH = 12):
    buffer_size = 50 # 50
    min_in_filterEachbuffer = []
    VecThresh = [0 for x in range(buffer_size - windowSize - filter_size )]
    timeWindVec = []
    for index in range(buffer_size, len(VDC_Slow)):
        timeWind = timeS[index]
        timeWindVec.append(timeWind)
        ind = index - windowSize - filter_size
        buffer1 = VDC_Slow[ind:ind + buffer_size]
        window1 = buffer1[-windowSize:]
        filter1 = buffer1[-(windowSize+filter_size): -windowSize]
        min_in_filter = np.mean(filter1)
        min_in_filterEachbuffer += [min_in_filter]
        SortWind = numpy.sort(window1)
        SortWindFlip = SortWind
        if len(SortWindFlip) > samTH:
            K_sampValWin = SortWindFlip[samTH-1]
            VecThresh += [K_sampValWin - min_in_filter]
        if (len(SortWindFlip) <= samTH) & (SortWindFlip.size > 0):
            K_sampValWin = SortWindFlip[0]
            VecThresh += [K_sampValWin - min_in_filter]
        index += 1
    VecThresh = [0 if x > 0 else float(x) for x in VecThresh]
    return np.abs(VecThresh),timeWindVec

def stage1_energy_calc50(SignalAfterDownsample ,TimeAfterDownsample,FsAfterDownsample, fIf = 6000,alpha = 0.2857,TH=12,windowSize=20,filter_size=15,samTH=12):
    ARC_MAX_BUFFER_SIZE = 50
    Signal = np.array(SignalAfterDownsample, dtype=float)
    timeDow = np.array(TimeAfterDownsample, dtype=float)
    DClevel = 0# 8200
    DcLevelCounter = 0
    DCLevelSum = 0
    PlaceInBit = 0
    arcbw = 50
    SamplesPerBitF = math.floor(FsAfterDownsample / arcbw)
    # build Hamming

    HammWindow = []
    for i in range(SamplesPerBitF):
        val = 0.54 - 0.46 * np.cos(2 * np.pi * i / SamplesPerBitF)
        HammWindow.append(val)
    # build dft vectors

    EnergyReSum = 0
    EnergyImSum = 0
    LastEnergy = 0
    EnergyVec = []
    m = 0
    CurrentIndex = 0
    count = 0
    After_First_buffer = False
    ArcDetectStage1 = False
    # LastEnergy = 0
    Buffer = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
              0, 0,
              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ArcInd = []

    # start

    for index, value in enumerate(Signal):
        DCLevelSum += value
        sample = value - DClevel
        DcLevelCounter += 1
        if DcLevelCounter == SamplesPerBitF: # to check
            DClevel = DCLevelSum / DcLevelCounter
            DCLevelSum = 0
            DcLevelCounter = 0
        PlaceInBit += 1
        if PlaceInBit >= SamplesPerBitF:  # each 357 do calc
            PlaceInBit -= SamplesPerBitF
            Energy = np.square(EnergyReSum) + np.square(EnergyImSum)
            EnergyReSum = 0
            EnergyImSum = 0
            EnergyIIR = alpha * Energy + (1 - alpha) * LastEnergy
            EnergyVec.append(EnergyIIR)
            LastEnergy = EnergyIIR
            Buffer[CurrentIndex] = EnergyIIR
            CurrentIndex += 1
            if CurrentIndex == ARC_MAX_BUFFER_SIZE - 1:
                CurrentIndex = 0
                After_First_buffer = True
            if After_First_buffer:
                Filter = Buffer[:filter_size]
                window = Buffer[filter_size:filter_size + windowSize]
                min_in_filter = min(Filter)
                m += 1
                for val_in_window in window:
                    if val_in_window - TH > min_in_filter:
                        count += 1
                    if count > samTH:
                        ArcDetectStage1 = True
        else:
            sample_after_window = HammWindow[PlaceInBit] * sample
            EnergyImSum += (sample_after_window * np.sin(2 * np.pi * fIf * PlaceInBit / FsAfterDownsample))
            EnergyReSum += (sample_after_window * np.cos(2 * np.pi * fIf * PlaceInBit / FsAfterDownsample))

    # timeEnergy = timeDow[::SamplesPerBitF]
    # timeEnergy = timeEnergy[0:len(EnergyVec)]
    #log_energy1 = 10 * np.log10(EnergyVec)
    log_energy1 = 10 * np.log10([e / 100000 for e in EnergyVec])
    log_energy2 = np.insert(log_energy1, 0, 0)
    log_energy = log_energy2[2:]
    # timeEnergy = timeEnergy[0:len(log_energy)]
    timeEnergy2 = np.arange(0, timeDow[-1] + 1 / arcbw, 1 / arcbw)
    timeEnergy = timeEnergy2[2:]
    return log_energy,timeEnergy

def stage1_energy_calc50_zcr_noam(signal, fs, zero_crossings, fif=6000, alpha=0.2121, th=12, window_size=20, filter_size=15, sam_th=12):
    signal = np.array(signal, dtype=float)
    hamm_window = [0.54 - 0.46 * np.cos(2 * np.pi * x / 333) for x in range(334)]
    samples_per_bit = np.append(np.insert(np.diff(zero_crossings), 0, zero_crossings[0]),
                                len(signal) - zero_crossings[-1])
    buffer = [0] * 50

    energy_im_sum, energy_re_sum, dc_level_sum, dc_level, dc_level_counter, place_in_bit = 0, 0, 0, 0, 0, 0
    energy_vec, last_energy, current_index, k = [], 0, 0, 1

    for value in signal[zero_crossings[0]:zero_crossings[-1]]:
        dc_level_sum += value
        dc_level_counter += 1
        sample = (value - dc_level)
        if dc_level_counter == samples_per_bit[k]:
            dc_level = dc_level_sum / dc_level_counter
            dc_level_sum, dc_level_counter = 0, 0
        place_in_bit += 1
        if place_in_bit >= samples_per_bit[k]:
            place_in_bit, k = 0, k + 1
            energy = energy_re_sum**2 + energy_im_sum**2
            last_energy = alpha * energy + (1 - alpha) * last_energy
            energy_vec.append(last_energy)
            buffer[current_index] = last_energy
            current_index = 0 if current_index == 49 else current_index + 1
            if current_index == 49 and any(val - th > min(buffer[:filter_size]) for val in buffer[filter_size:]):
                arc_detect_stage1 = True
            energy_im_sum, energy_re_sum = 0, 0

        try:
            hamm= 0.54-  0.46 * np.cos( (2.0 * place_in_bit  * np.pi) / 333 )
            sample_after_window = hamm * sample
        except IndexError:
            sample_after_window = hamm_window[-1] * sample
        energy_im_sum += sample_after_window * np.sin(2 * np.pi * fif * place_in_bit / fs)
        energy_re_sum += sample_after_window * np.cos(2 * np.pi * fif * place_in_bit / fs)

    return 10 * np.log10([x / 100000 for x in energy_vec[1:]])

def stage1_energy_calc50_ZCR(SignalAfterDownsample ,TimeAfterDownsample,FsAfterDownsample, zero_crossingsIdx2,fIf = 6000,alpha = 0.2121,TH=12,windowSize=20,filter_size=15,samTH=12):
    ARC_MAX_BUFFER_SIZE = 50
    Signal = np.array(SignalAfterDownsample, dtype=float)
    DClevel = 0# 8200
    DcLevelCounter = 0
    DCLevelSum = 0
    PlaceInBit = 0
    SamplesPerBitFzcr = np.diff(zero_crossingsIdx2) #math.floor(FsAfterDownsample / arcbw)
    zcrlast = len(Signal) - zero_crossingsIdx2[-1]
    SamplesPerBitF11 = np.insert(SamplesPerBitFzcr,0,zero_crossingsIdx2[0]) #zero_crossingsIdx2[0]
    SamplesPerBitF = np.append(SamplesPerBitF11,zcrlast) #zero_crossingsIdx2[0]

    HammWindow = []
    for j in range(len(SamplesPerBitF)):
        HammWindowj = []
        for i in range(SamplesPerBitF[j]):
            val = 0.54 - 0.46 * np.cos(2 * np.pi * i / SamplesPerBitF[j])
            HammWindowj.append(val)
        HammWindow.append(HammWindowj)

    # build dft vectors

    EnergyReSum = 0
    EnergyImSum = 0
    LastEnergy = 0
    EnergyVec = []
    m = 0
    CurrentIndex = 0
    count = 0
    After_First_buffer = False
    ArcDetectStage1 = False
    # LastEnergy = 0
    Buffer = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
              0, 0,
              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ArcInd = []

    # timeDow = np.array(TimeAfterDownsample[zero_crossingsIdx2[0]:zero_crossingsIdx2[-1]], dtype=float)

    # start
    k = 1 # try k =1 until -1 and substract drom signal
    for index, value in enumerate(Signal[zero_crossingsIdx2[0]:zero_crossingsIdx2[-1]]):
        DCLevelSum += value
        sample = value - DClevel
        DcLevelCounter += 1
        if DcLevelCounter == SamplesPerBitF[k]: # to check
            DClevel = DCLevelSum / DcLevelCounter
            DCLevelSum = 0
            DcLevelCounter = 0
        PlaceInBit += 1
        if PlaceInBit >= SamplesPerBitF[k]:  # each 357 do calc
            PlaceInBit = 0
            k+=1
            Energy = np.square(EnergyReSum) + np.square(EnergyImSum)
            EnergyReSum = 0
            EnergyImSum = 0
            EnergyIIR = alpha * Energy + (1 - alpha) * LastEnergy
            EnergyVec.append(EnergyIIR)
            LastEnergy = EnergyIIR
            Buffer[CurrentIndex] = EnergyIIR
            CurrentIndex += 1
            if CurrentIndex == ARC_MAX_BUFFER_SIZE - 1:
                CurrentIndex = 0
                After_First_buffer = True
            if After_First_buffer:
                Filter = Buffer[:filter_size]
                window = Buffer[filter_size:filter_size + windowSize]
                min_in_filter = min(Filter)
                m += 1
                for val_in_window in window:
                    if val_in_window - TH > min_in_filter:
                        count += 1
                    if count > samTH:
                        ArcDetectStage1 = True
        else:
            sample_after_window = HammWindow[k][PlaceInBit] * sample
            EnergyImSum += (sample_after_window * np.sin(2 * np.pi * fIf * PlaceInBit / FsAfterDownsample))
            EnergyReSum += (sample_after_window * np.cos(2 * np.pi * fIf * PlaceInBit / FsAfterDownsample))


    log_energy1 = 10 * np.log10([e / 100000 for e in EnergyVec])
    timeDow = np.array(TimeAfterDownsample[zero_crossingsIdx2[1:]], dtype=float)
    timeE = timeDow
    timeEnergy = timeE[1:]
    log_energy = log_energy1[1:]

    return log_energy,timeEnergy

def stage1_energy_calc50_ZCR_hammingFixed(SignalAfterDownsample ,TimeAfterDownsample,FsAfterDownsample, zero_crossingsIdx2,fIf = 6000,alpha = 0.2121,TH=12,windowSize=20,filter_size=15,samTH=12):
    ARC_MAX_BUFFER_SIZE = 50
    Signal_zcr = np.array(SignalAfterDownsample[zero_crossingsIdx2[0]:zero_crossingsIdx2[-1]], dtype=float)
    DClevel = 0
    DcLevelCounter = 0
    DCLevelSum = 0
    PlaceInBit = 0
    SamplesPerBitFzcr = np.diff(zero_crossingsIdx2)
    SamplesPerBitF = SamplesPerBitFzcr

    #build fixed Hamming
    HammWindow = []
    for i in range(0,333):
        val = 0.54 - 0.46 * np.cos(2 * np.pi * i / 333)
        HammWindow.append(val)

    # build dft vectors

    EnergyReSum = 0
    EnergyImSum = 0
    LastEnergy = 0
    EnergyVec = []
    CurrentIndex = 0
    Buffer = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
              0, 0,
              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    # start
    k = 0
    for index, value in enumerate(Signal_zcr):
        DCLevelSum += value
        sample = value - DClevel
        DcLevelCounter += 1
        if DcLevelCounter == SamplesPerBitF[k]:
            DClevel = DCLevelSum / DcLevelCounter
            DCLevelSum = 0
            DcLevelCounter = 0
        PlaceInBit += 1
        if PlaceInBit >= SamplesPerBitF[k]:
            PlaceInBit = 0
            k+=1
            Energy = np.square(EnergyReSum) + np.square(EnergyImSum)
            EnergyReSum = 0
            EnergyImSum = 0
            EnergyIIR = alpha * Energy + (1 - alpha) * LastEnergy
            EnergyVec.append(EnergyIIR)
            LastEnergy = EnergyIIR
            Buffer[CurrentIndex] = EnergyIIR
            CurrentIndex += 1
            if CurrentIndex == ARC_MAX_BUFFER_SIZE - 1:
                CurrentIndex = 0
        else:
            try:
                sample_after_window = HammWindow[PlaceInBit] * sample
            except:
                sample_after_window = HammWindow[-1] * sample
            EnergyImSum += (sample_after_window * np.sin(2 * np.pi * fIf * PlaceInBit / FsAfterDownsample))
            EnergyReSum += (sample_after_window * np.cos(2 * np.pi * fIf * PlaceInBit / FsAfterDownsample))


    log_energy1 = 10 * np.log10([e / 100000 for e in EnergyVec])
    timeDow = np.array(TimeAfterDownsample[zero_crossingsIdx2[0:]], dtype=float)
    # timeDow = np.array(TimeAfterDownsample[zero_crossingsIdx2[0:-1]], dtype=float)
    timeE = timeDow
    timeEnergy = timeE#[1:]
    log_energy = log_energy1#[1:]

    return log_energy,timeEnergy

def zcr_calc(Iac_L1C):
    # ZCR Iac L1 - I.e., zero_crossings will contain the indices of elements before which a zero crossing occurs. If you want the elements after, just add 1 to that array.
    zero_crossings = numpy.where(numpy.diff(numpy.sign(Iac_L1C)))[0]  # .astype(int)
    zero_crossingsIdx = zero_crossings.astype(int)
    if Iac_L1C[zero_crossingsIdx[0] + 2] > 0:
        zero_crossingsIdx22 = np.array(zero_crossingsIdx[::2]).astype(int)  # [random_values.astype(int)]#zero_crossingsIdx[::2]
    else:
        zero_crossingsIdxP = zero_crossingsIdx[1:]
        zero_crossingsIdx22 = np.array(zero_crossingsIdxP[::2]).astype(int)
    DiffZCR = np.diff(zero_crossingsIdx22)>300 #zero_crossingsIdx2[np.diff(zero_crossingsIdx2)>200]
    zero_crossingsIdx2_1 = zero_crossingsIdx22[1:]
    zero_crossingsIdx2_lst = zero_crossingsIdx2_1[DiffZCR].tolist()
    zero_crossingsIdx2_lst.insert(0,zero_crossingsIdx22[0])
    zero_crossingsIdx2 = np.array(zero_crossingsIdx2_lst)
    return zero_crossingsIdx2

def zcr_calcByPlaceInBit(PlaceInBit):
    # ZCR Iac L1 - I.e., zero_crossings will contain the indices of elements before which a zero crossing occurs. If you want the elements after, just add 1 to that array.
    zero_crossings = numpy.where(np.array(PlaceInBit) == 0)[0]  # .astype(int)
    zero_crossingsIdx22 = zero_crossings.astype(int)
    DiffZCR = np.diff(zero_crossingsIdx22)>300 #zero_crossingsIdx2[np.diff(zero_crossingsIdx2)>200]
    zero_crossingsIdx2_1 = zero_crossingsIdx22[1:]
    zero_crossingsIdx2_lst = zero_crossingsIdx2_1[DiffZCR].tolist()
    zero_crossingsIdx2_lst.insert(0,zero_crossingsIdx22[0])
    zero_crossingsIdx2 = np.array(zero_crossingsIdx2_lst)

    return zero_crossingsIdx2
def butter_lowpass1(cutoff, fs, order):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    # b, a = signal.butter(order, normal_cutoff, btype='low', analog=True)
    return b, a

def newVdcProcessCounterZCR(VdcFast,time_RX_RAWC,zero_crossingsIdx2,cutoff,fs,order,windowSize = 10, filter_size=25, samTH = 2, TH_Vdc = 0.05):
    B, A = butter_lowpass1(cutoff, fs, order=order)
    b0 = B[0]
    b1 = B[1]
    a0 = A[0] # Almost always 1
    a1 = A[1]#(b0+b1) -1
    x_n_1 = VdcFast[0]
    Y_IIR_1 = x_n_1
    VdcVec = [Y_IIR_1]
    for index in range(0, int(len(VdcFast)-1)):
        x_n = VdcFast[index]
        Y_IIR = (b0 * x_n +b1 * x_n_1 - (a1) * Y_IIR_1)/a0
        VdcVec.append(Y_IIR)
        Y_IIR_1 = Y_IIR
        x_n_1 = x_n

    T = 1 / fs
    N_ = len(VdcVec)
    time_ = np.linspace(0, N_ * T, N_)
    VdcVecMean = []
    for i in range(0, len(zero_crossingsIdx2)-1):
        bufferVdc = VdcVec[zero_crossingsIdx2[i]:zero_crossingsIdx2[i+1]]
        MeanVdc = np.mean(bufferVdc)
        VdcVecMean.append(MeanVdc)


    time = np.array(time_RX_RAWC[zero_crossingsIdx2[0:]], dtype=float)
    # time = np.array(time_RX_RAWC[zero_crossingsIdx2[0:]], dtype=float)

    ##################################
    buffer_size = 50
    VdcCounterVector =  []
    timeCounterVecVdc = []
    for index in range(buffer_size, len(VdcVecMean)):
        VdcCounter = 0
        ind = index - buffer_size
        timeCounterVdc = time[index]
        timeCounterVecVdc.append(timeCounterVdc)
        # VDC_Slow Fill buff
        bufferV = VdcVecMean[ind:ind + buffer_size]
        windowV = bufferV[-windowSize:]
        filterV = bufferV[-(windowSize+filter_size): -windowSize]
        Min_in_filter_V = np.min(filterV)
        for value in windowV:
            if value < (Min_in_filter_V- TH_Vdc):
                VdcCounter+=1
        VdcCounterVector +=[VdcCounter]

    #####################################
    buffer_size = 50
    min_in_filterEachbuffer = []
    VecThresh = []
    timeWindVec = []
    for index in range(buffer_size, len(VdcVecMean)):
        timeWind = time[index]
        timeWindVec.append(timeWind)
        ind = index - buffer_size
        buffer1 = VdcVecMean[ind:ind + buffer_size]
        window1 = buffer1[-windowSize:]
        filter1 = buffer1[-(windowSize + filter_size): -windowSize]
        min_in_filter = np.min(filter1)
        min_in_filterEachbuffer += [min_in_filter]
        SortWind = numpy.sort(window1)
        SortWindFlip = SortWind
        if len(SortWindFlip) > samTH:
            K_sampValWin = SortWindFlip[samTH - 1]
            VecThresh += [K_sampValWin - min_in_filter]
        if (len(SortWindFlip) <= samTH) & (SortWindFlip.size > 0):
            K_sampValWin = SortWindFlip[0]
            VecThresh += [K_sampValWin - min_in_filter]
        index += 1
    VecThresh = [0 if x > 0 else float(x) for x in VecThresh]

    return VdcVec, time_,np.abs(VecThresh),timeWindVec , VdcCounterVector , timeCounterVecVdc,VdcVecMean,time

def IacProcessZCR(Iac,time_RX_RAWC,ZCRindx2):
    RmsIacVEC = []
    for i in range(0, len(ZCRindx2)-1):
        bufferIac = Iac[ZCRindx2[i]:ZCRindx2[i+1]]
        RmsIac = np.sqrt(np.mean(np.array(bufferIac) ** 2))
        RmsIacVEC.append(RmsIac)

    time = np.array(time_RX_RAWC[ZCRindx2[0:]], dtype=float)
    return RmsIacVEC, time

def filterEnergy(Energy,alpha):
    LastEnergy = 0
    EnergyFilteredVec1 = []
    for index, value in enumerate(Energy):
        EnergyIIR = alpha * value + (1 - alpha) * LastEnergy
        EnergyFilteredVec1.append(EnergyIIR)
        LastEnergy = EnergyIIR
    # /100000 log_energy1 = 10 * np.log10([e / 100000 for e in EnergyVec])
    EnergyFilteredVec = 10 * np.log10([e / 100000 for e in EnergyFilteredVec1])
    return EnergyFilteredVec








# folder = r"M:\Users\LinoyL\DATA_ARCS\ALL_DATA\NewDATA_MSP\S5"
folder = r"V:\HW_Infrastructure\Analog_Team\ARC_Data\Results\Jupiter\Jupiter+ Improved (7E0872F4-EC)\Arc Interrupt 50Hz\Test 15 - FW 22.01 test 02"


NumRec = 1500
listFolder = [x[0] for x in os.walk(folder)]
Fs = 35
T = 1 / Fs
Fs_G = 50
T_G = 1 / Fs_G
Fs_SPI = 16667

T_SPI = 1 / Fs_SPI
### eddy take these parameters
ContTH = 3 #2
THVdc = 0.043 #0.045
W_Size = 20 #7
F_Size = 15 #28
CutOff_fs = 1.45#2.5
########
# ContTH = 2
# THVdc = 0.045
# W_Size = 7
# F_Size = 28
# CutOff_fs = 2.5


n=0
a_filter_50 = 0.2119
TH_Energy50 = 11
Counter_Energy50 = 12


plots_per_pane = 5
fig = initialize_fig(row=3, col=1, plots_per_pane=plots_per_pane, shared_xaxes=True,
                     subplot_titles=['Rx Raw ~16.6 kHz','Energy 50 Hz','Vdc Slow 50 Hz'])

file_arr_list=list(range(1, NumRec))
for i in range(0,len(listFolder)):
    for k, filenameSPI in enumerate([f for f in os.listdir(listFolder[i]) if f.endswith('.txt') and 'spi' in f]):
        filename1SPI = f"{listFolder[i]}/{filenameSPI}"
        NameSPI = filenameSPI[:-4]
        splitFile = NameSPI.split()
        StringFileSPI = ' '.join(map(str, splitFile[:2]))
        StringFile = StringFileSPI
        # RX_RAW50, Vdc_fast50, EnergyLast, RmsIac, VdcS, PlaceInBit = SPI_Reading.read_file(filename1SPI)
        RX_RAW50, Vdc_fast50, VdcS, VdcCounter, PlaceInBit, EnergyBuffer = SPI_Reading.read_file(filename1SPI)
        RmsIac = VdcS
        #  --ZCR-- by PlaceInBit
        zero_crossingsIdx22 = zcr_calcByPlaceInBit(PlaceInBit)

        Vdc_fast1 = Vdc_fast50[:zero_crossingsIdx22[-1]]
        RX_RAW = RX_RAW50[:zero_crossingsIdx22[-1]]
        indicesStop1 = cut_indxS(RX_RAW, Vdc_fast1, Fs_SPI)
        indicesStop = indicesStop1 #+ 2700
        Vdc_fast2 = Vdc_fast1[:indicesStop]
        RX_RAWC = RX_RAW[:indicesStop]
        if zero_crossingsIdx22[-1] >= indicesStop:
            zero_crossingsIdx2 = zero_crossingsIdx22[zero_crossingsIdx22<indicesStop]
        else:
            zero_crossingsIdx2 = zero_crossingsIdx22


        L = len(RX_RAWC)
        time_RX_RAWC = np.linspace(0, L * T_SPI, L)

        Vdc_fast = [x for x in Vdc_fast2]
        Vdc_fastRemoveDC = Vdc_fast
        N = len(Vdc_fast)

        ## 50 Hz ##
        #### PWR Energy and VDC
        # Energy and VDC ZCR new 50Hz by SPI directly (~PWR)
        # energiesPWR50_1 = np.array(EnergyLast)[zero_crossingsIdx2]#/100000
        # energiesPWR50 = filterEnergy(energiesPWR50_1, a_filter_50)
        energiesPWR50 = filterEnergy(np.array(EnergyBuffer)[zero_crossingsIdx2], 1)
        VdcS_PWR = np.array(VdcS)[zero_crossingsIdx2] #VdcS
        L_PWR50 = len(energiesPWR50)
        TimePWR = np.linspace(0, L_PWR50 * T_G, L_PWR50)

        # # Iac ZCR RMS 50Hz  by SPI directly
        # RmsIacC = np.array(RmsIac)[zero_crossingsIdx2]

        # Energy ZCR new 50Hz by RX
        log_energy50, timeEnergy50 = stage1_energy_calc50_ZCR_hammingFixed(RX_RAWC, time_RX_RAWC, Fs_SPI,zero_crossingsIdx2, fIf=6000, alpha=a_filter_50, TH=TH_Energy50,
                                                    windowSize=20, filter_size=15, samTH=Counter_Energy50)
        # ClassifyEnergy50 = labelingByRX_Energy50(RX_RAWC, time_RX_RAWC, log_energy50, timeEnergy50)
        # VecThreshE50, timeTH_E50 = energy_rise_calcCorrect(log_energy50, timeEnergy50, windowSize=20, filter_size=15, samTH=Counter_Energy50)

        # Vdc ZCR new 50Hz
        VdcVecFileredNew, timeNew, VecThreshV_NN, timeWindVec_NN, VdcCounterVector, timeCounterVecVdc,Vdc_Slow_New,time_Vdc_Slow_New = newVdcProcessCounterZCR(
            Vdc_fast,time_RX_RAWC,zero_crossingsIdx2, CutOff_fs, Fs_SPI, 1, windowSize=W_Size, filter_size=F_Size, samTH=ContTH, TH_Vdc=THVdc)


        # Detection E 50 Hz
        EnergyCounterVector50, timeCounterVec50 = Counter_Energy_calcCorrect(log_energy50, timeEnergy50, buffer_size=50, windowSize=20, filter_size=15, TH_Energy=TH_Energy50)
        # Detection I 50 Hz
        # time_RmsIacC = timeEnergy50
        # # IacCounterVector,timeCounterVecIac = Counter_Iac_calcCorrect(RmsIacC, time_RmsIacC, buffer_size=50, windowSize=20, filter_size=15, TH_Iac=0.2)
        # Binary Detection
        EnergyCounterVectorBinary = np.where(np.array(EnergyCounterVector50) >= Counter_Energy50, 1, 0)
        # IacCounterVectorBinary = np.where(np.array(IacCounterVector) >= 12, 1, 0)
        VdcCounterVectorBinary = np.where(np.array(VdcCounterVector) >= ContTH, 1, 0)


        fig.add_trace(go.Scattergl(x=time_RX_RAWC,
                                   y=RX_RAWC,
                                   name='RX_RAW' + 'Rec' + StringFile, mode="lines",
                                   visible=False,
                                   showlegend=True),
                      row=1, col=1, secondary_y=False)

        fig.add_trace(go.Scattergl(x=timeEnergy50,
                                   y=log_energy50,
                                   name='Energy Offline 50 Hz ' + 'Rec ' + StringFile, mode="lines",
                                   visible=False,
                                   showlegend=True),
                      row=2, col=1, secondary_y=False)
        fig.add_trace(go.Scattergl(x=TimePWR,
                                   y=energiesPWR50[2:],
                                   name='Energy PWR 50 Hz ' + 'Rec ' + StringFile, mode="lines",
                                   visible=False,
                                   showlegend=True),
                      row=2, col=1, secondary_y=False)        # energiesPWR50  VdcS_PWR  TimePWR

        fig.add_trace(go.Scattergl(x=time_Vdc_Slow_New,
                                   y=Vdc_Slow_New,
                                   name='Vdc Slow Offline 50 Hz ' + 'Rec ' + StringFile, mode="lines",
                                   visible=False,
                                   showlegend=True),
                      row=3, col=1, secondary_y=False)
        fig.add_trace(go.Scattergl(x=time_Vdc_Slow_New,
                                   y=VdcS_PWR[1:],
                                   name='Vdc Slow PWR 50 Hz ' + 'Rec ' + StringFile, mode="lines",
                                   visible=False,
                                   showlegend=True),
                      row=3, col=1, secondary_y=False)
        # fig.add_trace(go.Scattergl(x=time_RmsIacC,
        #                            y=RmsIacC,
        #                            name='Iac RMS 50 Hz ' + 'Rec ' + StringFile, mode="lines",
        #                            visible=False,
        #                            showlegend=True),
        #               row=4, col=1, secondary_y=False)
        # fig.add_trace(go.Scattergl(x=timeCounterVec50,
        #                            y=EnergyCounterVectorBinary,
        #                            name='Binary Detection Counter Energy 50 Hz ' + 'Rec ' + StringFile, mode="lines",
        #                            visible=False,
        #                            showlegend=True),
        #               row=5, col=1, secondary_y=False)
        #
        # fig.add_trace(go.Scattergl(x=timeCounterVecVdc,
        #                            y=VdcCounterVectorBinary,
        #                            name='Binary Detection Counter Vdc Slow 50 Hz ' + 'Rec ' + StringFile, mode="lines",
        #                            visible=False,
        #                            showlegend=True),
        #               row=6, col=1, secondary_y=False)
        #
        # fig.add_trace(go.Scattergl(x=timeCounterVecIac,
        #                            y=IacCounterVectorBinary,
        #                            name='Binary Detection Counter Iac RMS 50 Hz ' + 'Rec ' + StringFile, mode="lines",
        #                            visible=False,
        #                            showlegend=True),
        #               row=7, col=1, secondary_y=False)





for i in range(plots_per_pane):
    fig.data[i].visible = True
steps = []
for i in range(0, int(len(fig.data) / plots_per_pane)):
    Temp = file_arr_list[i]

    step = dict(
        method="update",
        args=[{"visible": [False] * len(fig.data)},
              {"title": "Slider  switched to step "}],
        label=str(Temp)  # layout attribute
    )
    j = i * plots_per_pane
    for k in range(plots_per_pane):
        step["args"][0]["visible"][j + k] = True
    steps.append(step)
sliders = [dict(
    active=10,
    currentvalue={"prefix": "REC: "},
    pad={"t": 50},
    steps=steps
)]
fig.update_layout(
    sliders=sliders)

config = {'scrollZoom': True, 'responsive': False, 'editable': True, 'modeBarButtonsToAdd': ['drawline',
                                                                                             'drawopenpath',
                                                                                             'drawclosedpath',
                                                                                             'drawcircle',
                                                                                             'drawrect',
                                                                                             'eraseshape'
                                                                                             ]}


fig.write_html(' PWR&SPI E and V.HTML', auto_open=True, config=config)


