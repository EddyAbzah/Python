import numpy
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
# import SPI_extraction
import SPI_Reading
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
import glob
import math
import time
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from scipy import signal
from warnings import simplefilter
from scipy.fft import fft, rfft
from scipy.fft import fftfreq, rfftfreq
import plotly.graph_objs as go
import heapq
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import Jup_pwr_vdc_criteria




def initialize_fig(row = 4,col = 1,plots_per_pane=4,shared_xaxes=True,subplot_titles=['Rx','Rx','Rx','Rx']):

    all_specs = np.array([[{"secondary_y": True}] for x in range((row*col))])
    all_specs_reshaped = (np.reshape(all_specs, (col, row)).T).tolist()
    fig= make_subplots(rows=row, cols=col,specs=all_specs_reshaped, shared_xaxes=shared_xaxes,   subplot_titles=subplot_titles)

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


def notch_filter(fs,f0,Q):
    #fs = 200.0  # Sample frequency (Hz)
    #f0 = 60.0  # Frequency to be removed from signal (Hz)
    #Q = 30.0  # Quality factor
    w0 = f0 / (fs / 2)  # Normalized Frequency
    b, a = signal.iirnotch(w0, Q)
    # Look at frequency response
    w, h = signal.freqz(b, a)
    freq = w * fs / (2 * np.pi)
    plt.plot(freq, 20 * np.log10(abs(h)))


def FFT(x,Fs):
    fft1 = np.abs(np.fft.fftshift(np.fft.fft(x)))
    #fft1 = fft1[len(fft1) // 2:]
    N = len(x)
    normalize = N / 4
    fourier = fft1#(x)
    norm_amplitude = fourier / normalize # np.abs(fourier) / normalize
    norm_amplitude = norm_amplitude[len(norm_amplitude) // 2:]

    sampling_rate = Fs # It's used as a sample spacing
    frequency_axis = fftfreq(N, d=1.0 / sampling_rate)
    frequency_axis = frequency_axis[:len(frequency_axis) // 2]
    #frequencies = np.arange(len(fft))
    # fft = np.abs(np.fft.fftshift(np.fft.fft(x)))
    # fft = fft[fft.shape[0] // 2:]
    return norm_amplitude,frequency_axis

def butter_lowpass1(cutoff, fs, order):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter1(data, cutoff, fs, order):
    #data1 = data - data[0]
    # data1 = [x - 760 for x in data]
    #data1 = [x - data[0] for x in data]
    b, a = butter_lowpass1(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y


def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y


def LPF_eachBuffer(Vdc_fast_With_arc,Fs_SPI,time_spi_fast,orderFilter = 1,fc = 10):
    # FilteredSignalLP = butter_lowpass_filter1(Vdc_fast_With_arc, 10, Fs_SPI, order=1)  # low pass
    # FilteredSignalLP = butter_lowpass_filter1(Vdc_fast_With_arc, fc, Fs_SPI, order=orderFilter)  # low pass
    Window_size = math.ceil(Fs_SPI/50)
    FilteredSignalLP_Vec = []#[0 for x in range(buffer_size - windowSize - filter_size)]
    timeVDC_SlowVec = []
    for index in range(0, len(Vdc_fast_With_arc),Window_size):
        timeSlow = time_spi_fast[index]
        timeVDC_SlowVec.append(timeSlow)
        ind = index
        Vdc_Window = Vdc_fast_With_arc[ind:ind + Window_size]
        #Vdc_Window = [x - Vdc_fast_With_arc[index] for x in Vdc_Window1]
        FilteredSignalLP = butter_lowpass_filter1(Vdc_Window, fc, Fs_SPI, order=orderFilter)  # low pass
        FilteredSignalLP_Vec.extend(FilteredSignalLP)
    return FilteredSignalLP_Vec


def LPF_autocorrelation_eachBuffer(Vdc_fast_With_arc,Fs_SPI,time_spi_fast,orderFilter ,fc ):
    # FilteredSignalLP = butter_lowpass_filter1(Vdc_fast_With_arc, 10, Fs_SPI, order=1)  # low pass
    # FilteredSignalLP = butter_lowpass_filter1(Vdc_fast_With_arc, fc, Fs_SPI, order=orderFilter)  # low pass
    #EpsBoundryCeil  = 0.05 * np.sqrt(np.mean(Vdc_fast_With_arc**2))
    EpsBoundryfloor =  np.sqrt(np.mean(np.array(Vdc_fast_With_arc[:61668]) ** 2)) * 0.9985 #0.7 #np.sqrt(np.mean(np.array(Vdc_fast_With_arc[:50000]) ** 2)) #- 0.15 * np.sqrt(np.mean(np.array(Vdc_fast_With_arc) ** 2))
    Window_size = math.ceil(Fs_SPI/50)
    FilteredSignalLP_Vec = []#[0 for x in range(buffer_size - windowSize - filter_size)]
    timeVDC_SlowVec = []
    ExccedEpsIntervalCounter = 0
    ExccedEpsIntervalCounterVec = []
    Vdc_WindowAcorrVec =[]
    Vdc_WindowAcorrSUMVec =[]
    BinnaryDetectionRes = []
    EpsBoundryfloorVec = []
    for index in range(0, len(Vdc_fast_With_arc),Window_size):
        #Vdc_Calibrate = [x - Vdc_fast_With_arc[index] for x in Vdc_fast_With_arc]
        timeSlow = time_spi_fast[index]
        ExccedEpsIntervalCounter = 0
        timeVDC_SlowVec.append(timeSlow)
        ind = index
        Vdc_Window = Vdc_fast_With_arc[ind:ind + Window_size]
        # Mean
        mean = numpy.mean(Vdc_Window)
        # Variance
        var = numpy.var(Vdc_Window)
        # Normalized data
        ndata = np.array(Vdc_Window) - mean
        acorr = numpy.correlate(ndata, ndata, 'full')[len(ndata) - 1:]
        acorr = acorr / var / len(ndata)
        acorrSUM = np.var(acorr / var / len(ndata))
        acorrList = acorr.tolist()
        acorrListSUM = acorrSUM.tolist()
        #Vdc_WindowRMS = np.sqrt(np.mean(np.array(Vdc_Window) ** 2)) * np.ones(len(Vdc_Window[:])) #* 0.998
        #Vdc_Window = [x - Vdc_fast_With_arc[index] for x in Vdc_Window1]
        Vdc_WindowVAR = np.var(Vdc_Window)
        Vdc_WindowAcorrVec.extend(acorrList)
        Vdc_WindowAcorrSUMVec.extend([acorrListSUM])
        FilteredSignalLP = butter_lowpass_filter1(Vdc_Window, fc, Fs_SPI, order=orderFilter)  # low pass
        FilteredSignalLP_Vec.extend(FilteredSignalLP)
        #EpsBoundryfloor = EpsBoundryfloorW* np.ones(len(FilteredSignalLP[:]))
        EpsBoundryfloorVec.extend(EpsBoundryfloor* np.ones(len(FilteredSignalLP[:])))
        for value in FilteredSignalLP:
            if value < EpsBoundryfloor:
                ExccedEpsIntervalCounter += 1
        ExccedEpsIntervalCounterVec.append(ExccedEpsIntervalCounter)
    #BinnaryDetectionResvec = np.zeros(len(ExccedEpsIntervalCounterVec))
    TH = 100
    for value in ExccedEpsIntervalCounterVec:
        if value > TH:
            BinnaryDetectionRes.append(1)
        else:
            BinnaryDetectionRes.append(0)
    # FilteredSignalLP_Vec.extend(FilteredSignalLP)
    return FilteredSignalLP_Vec , EpsBoundryfloorVec , ExccedEpsIntervalCounterVec , BinnaryDetectionRes,timeVDC_SlowVec,Vdc_WindowAcorrVec,Vdc_WindowAcorrSUMVec

def LPF_EpsilonInterval_eachBuffer(Vdc_fast_With_arc,Fs_SPI,time_spi_fast,orderFilter ,fc ):
    # FilteredSignalLP = butter_lowpass_filter1(Vdc_fast_With_arc, 10, Fs_SPI, order=1)  # low pass
    # FilteredSignalLP = butter_lowpass_filter1(Vdc_fast_With_arc, fc, Fs_SPI, order=orderFilter)  # low pass
    #EpsBoundryCeil  = 0.05 * np.sqrt(np.mean(Vdc_fast_With_arc**2))
    EpsBoundryfloor =  np.sqrt(np.mean(np.array(Vdc_fast_With_arc[:61668]) ** 2)) * 0.9986 #0.7 #np.sqrt(np.mean(np.array(Vdc_fast_With_arc[:50000]) ** 2)) #- 0.15 * np.sqrt(np.mean(np.array(Vdc_fast_With_arc) ** 2))
    Window_size = math.ceil(Fs_SPI/50)
    FilteredSignalLP_Vec = []#[0 for x in range(buffer_size - windowSize - filter_size)]
    timeVDC_SlowVec = []
    ExccedEpsIntervalCounter = 0
    ExccedEpsIntervalCounterVec = []
    Vdc_WindowVARVec =[]
    BinnaryDetectionRes = []
    EpsBoundryfloorVec = []
    for index in range(0, len(Vdc_fast_With_arc),Window_size):
        #Vdc_Calibrate = [x - Vdc_fast_With_arc[index] for x in Vdc_fast_With_arc]
        timeSlow = time_spi_fast[index]
        ExccedEpsIntervalCounter = 0
        timeVDC_SlowVec.append(timeSlow)
        ind = index
        Vdc_Window = Vdc_fast_With_arc[ind:ind + Window_size]
        #Vdc_WindowRMS = np.sqrt(np.mean(np.array(Vdc_Window) ** 2)) * np.ones(len(Vdc_Window[:])) #* 0.998
        #Vdc_Window = [x - Vdc_fast_With_arc[index] for x in Vdc_Window1]
        Vdc_WindowVAR = np.var(Vdc_Window)
        Vdc_WindowVARVec.append(Vdc_WindowVAR)
        FilteredSignalLP = butter_lowpass_filter1(Vdc_Window, fc, Fs_SPI, order=orderFilter)  # low pass
        FilteredSignalLP_Vec.extend(FilteredSignalLP)
        #EpsBoundryfloor = EpsBoundryfloorW* np.ones(len(FilteredSignalLP[:]))
        EpsBoundryfloorVec.extend(EpsBoundryfloor* np.ones(len(FilteredSignalLP[:])))
        for value in FilteredSignalLP:
            if value < EpsBoundryfloor:
                ExccedEpsIntervalCounter += 1
        ExccedEpsIntervalCounterVec.append(ExccedEpsIntervalCounter)
    #BinnaryDetectionResvec = np.zeros(len(ExccedEpsIntervalCounterVec))
    TH = 100
    for value in ExccedEpsIntervalCounterVec:
        if value > TH:
            BinnaryDetectionRes.append(1)
        else:
            BinnaryDetectionRes.append(0)
    # FilteredSignalLP_Vec.extend(FilteredSignalLP)
    return FilteredSignalLP_Vec , EpsBoundryfloorVec , ExccedEpsIntervalCounterVec , BinnaryDetectionRes,timeVDC_SlowVec , Vdc_WindowVARVec

def VdcProcess(Vdc,FsAfterDownsample): # current method
    arcbw = 50
    #GridFS = 50
    Mean_Size = int(FsAfterDownsample / arcbw)
    alpha = 1 - 1e-4
    # Mean_Size = 467
    LastVdc = 0
    VdcVec = []
    for index, val in enumerate(Vdc): # NOT really happen by alon zohar
        VdcIIR = alpha * val + (1 - alpha) * LastVdc
        VdcVec.append(VdcIIR)
        LastVdc = VdcIIR

    VdcVecMean = []
    for index in range(0, len(VdcVec) - Mean_Size, Mean_Size):
        bufferVdc = VdcVec[index:index + Mean_Size]
        MeanVdc = np.mean(bufferVdc)
        VdcVecMean.append(MeanVdc)
    Fs = arcbw# !!!!!!!!!!!!!!!!!!!!35
    T = 1 / Fs
    N = len(VdcVecMean)
    time = np.linspace(0, N * T, N)

    return VdcVecMean, time


def Vdc_Criteria(Vdc_fast_With_arc,Fs_SPI,time_spi_fast,Name,orderFilter ,fc ):
    Window_size = math.ceil(Fs_SPI/50)
    timeVDC_SlowVec = []
    FilteredSignalLPInitalize_Vec = []
    for index in range(0, int(Window_size*2) ,int(Window_size)):
        ind = index
        Vdc_Window = Vdc_fast_With_arc[ind:ind + Window_size]
        FilteredSignalLPInitalize = butter_lowpass_filter1(Vdc_Window, fc, Fs_SPI, order=orderFilter)  # low pass
        FilteredSignalLPInitalize_Vec.extend(FilteredSignalLPInitalize)

    FilteredSignalLPInitalize_VecCUT = FilteredSignalLPInitalize_Vec[int(Window_size / 2):int(Window_size*1.2)]
    FirstValMin = min(FilteredSignalLPInitalize_VecCUT)
    FirstIndxMin = FilteredSignalLPInitalize_VecCUT.index(FirstValMin) + int(Window_size / 2)

    ValVdc_WideWindowMinVec = []
    ValVdc_WideWindowMinVec2 = []
    IndxVdc_WideWindowMinVec = [int(FirstIndxMin)]
    FilteredSignalLP_Vec = []
    time_FilteredSignalLP_Vec = time_spi_fast[FirstIndxMin:]
    lowerboundryVec = []
    ExccedEpsIntervalCounter = 0
    ExccedEpsIntervalCounterVec = []
    n = 0
    MayArc = 0
    Arc = 0
    DetectionResVec = []
    DetectionResVecIndx = []
    movePointIndx = 0 #int(FirstIndxMin)
    movePointIndxVec = []
    movePointC = 1
    finishStep1 = 0
    finishStep2 = 0
    # Extract locals minimums per cycle
    for index in range( int(FirstIndxMin), len(Vdc_fast_With_arc),int(Window_size)):
        timeSlow = time_spi_fast[index]
        timeVDC_SlowVec.append(timeSlow)
        ind = index
        n = n + 1
        Vdc_WideWindow = Vdc_fast_With_arc[ind:ind + Window_size]
        Vdc_WideWindowFiltered = butter_lowpass_filter1(Vdc_WideWindow, fc, Fs_SPI, order=orderFilter)  # low pass
        Vdc_WideWindowFilteredList = Vdc_WideWindowFiltered.tolist()
        if index == FirstIndxMin:
            ValVdc_WideWindowMinVec2 = [Vdc_WideWindowFilteredList[0]]
            ValVdc_WideWindowMinVec = [Vdc_WideWindowFilteredList[0]]
        try:
            ValVdc_WideWindowMin = min(Vdc_WideWindowFilteredList[int(Window_size/2):])
            IndxVdc_WideWindowMin = Vdc_WideWindowFilteredList.index(ValVdc_WideWindowMin) #+ 35
        except:
            ValVdc_WideWindowMin = min(Vdc_WideWindowFilteredList)
            IndxVdc_WideWindowMin = Vdc_WideWindowFilteredList.index(ValVdc_WideWindowMin)
            print("Problem " + Name)
        #MinIndex = index + int(IndxVdc_WideWindowMin)

        movePointC = movePointC + 1
        ValVdc_WideWindowMinVec.append(ValVdc_WideWindowMin)
        ValVdc_WideWindowMinVec2.append(ValVdc_WideWindowMin)
        IndxVdc_WideWindowMinVec.append(index +  int(IndxVdc_WideWindowMin))
        FilteredSignalLP_Vec.extend(Vdc_WideWindowFilteredList)
        ExccedEpsIntervalCounter = 0
        # if int (movePointC - 3) ==0 or (int (movePointC - 4)%3 ==0 and int (movePointC - 4)%4 !=0):
        #     movePointIndx = index +  int(IndxVdc_WideWindowMin)
        #     movePointIndxVec.extend([movePointIndx])
        #

        # if (len(ValVdc_WideWindowMinVec2)%7 == 0) and (n ==4): # to try modulo 3 or to check raise also maybe
        if (finishStep2 == 0) and (n == 4):  # to try modulo 3 or to check raise also maybe
            second2CycleRelevant = FilteredSignalLP_Vec[int(-2*Window_size):]
            for value in second2CycleRelevant:
            #for value in FilteredSignalLP_Vec[int(IndxVdc_WideWindowMinVec[-2]):]:
                if value < lowerboundry:
                    ExccedEpsIntervalCounter += 1
            if ExccedEpsIntervalCounter <= int(ExccedEpsIntervalCounterVec[-1]*0.5) and MayArc == 0.5: # if is arc i expect that the ExccedEpsIntervalCounter here will decrese (lower) because the vdc suposos come back to the baseline
                Arc = 1
                print("ARC Detection in index : " + str(index+Window_size))
            else:
                Arc = 0
            finishStep2 = 1
            ExccedEpsIntervalCounterVec.append(ExccedEpsIntervalCounter)
            DetectionResVec.extend([Arc])
            DetectionResVecIndx.extend([int(index+Window_size)])


        # if (len(ValVdc_WideWindowMinVec2)%5 == 0) and (n ==2): # to try modulo 3 or to check raise also maybe
        if (finishStep1 == 1) and (n == 2):  # to try modulo 3 or to check raise also maybe
            first2CycleRelevant = FilteredSignalLP_Vec[int(-2*Window_size):]
            # for value in FilteredSignalLP_Vec[int(IndxVdc_WideWindowMinVec[-2]):]:
            for value in first2CycleRelevant:
                if value < lowerboundry:
                    ExccedEpsIntervalCounter += 1
            if ExccedEpsIntervalCounter >=100:
                MayArc = 0.5
            else:
                MayArc = 0
            finishStep1 = 0
            ExccedEpsIntervalCounterVec.append(ExccedEpsIntervalCounter)
            DetectionResVec.extend([MayArc])
            DetectionResVecIndx.extend([int(index+Window_size)])



        # if ((len(ValVdc_WideWindowMinVec2)%3 == 0) and n==6) or (int(len(ValVdc_WideWindowMinVec2)-3) == 0): # to try modulo 3 or to check raise also maybe
        if ((finishStep2 == 1) and n == 6) or (int(len(ValVdc_WideWindowMinVec2) - 3) == 0):  # to try modulo 3 or to check raise also maybe
            if (finishStep2 == 1) :
                MeanofMins = np.mean(np.array(ValVdc_WideWindowMinVec2[-5:-2]))
                STDofMins = np.std(np.array(ValVdc_WideWindowMinVec2[-5:-2]))
                lowerboundry = MeanofMins - 0.5 * STDofMins
                first2CycleRelevant = FilteredSignalLP_Vec[int(-2 * Window_size):]
                for value in first2CycleRelevant:
                    if value < lowerboundry:
                        ExccedEpsIntervalCounter += 1
                if ExccedEpsIntervalCounter >= 100:
                    MayArc = 0.5
                else:
                    MayArc = 0
                finishStep1 = 0
                finishStep2 = 0
                n = 2
                ExccedEpsIntervalCounterVec.append(ExccedEpsIntervalCounter)
                DetectionResVec.extend([MayArc])
                DetectionResVecIndx.extend([int(index + Window_size)])
            else:
                MeanofMins = np.mean(np.array(ValVdc_WideWindowMinVec2[-3:]))
                STDofMins = np.std(np.array(ValVdc_WideWindowMinVec2[-3:]))
                lowerboundry = MeanofMins -0.5*STDofMins
                n = 0
                finishStep1 = 1
            lowerboundryVec.extend([lowerboundry])

    return FilteredSignalLP_Vec ,time_FilteredSignalLP_Vec , ValVdc_WideWindowMinVec, IndxVdc_WideWindowMinVec , lowerboundryVec , ExccedEpsIntervalCounterVec , movePointIndxVec ,DetectionResVec

def minimumDetection(Vdc_fast_With_arc,Fs_SPI,time_spi_fast,Name,orderFilter ,fc ):
    #EpsBoundryfloor =  np.sqrt(np.mean(np.array(Vdc_fast_With_arc[:61668]) ** 2)) * 0.9986 #0.7 #np.sqrt(np.mean(np.array(Vdc_fast_With_arc[:50000]) ** 2)) #- 0.15 * np.sqrt(np.mean(np.array(Vdc_fast_With_arc) ** 2))
    #ExccedEpsIntervalCounterVec = []
    #Vdc_WindowVARVec =[]
    #BinnaryDetectionRes = []
    #EpsBoundryfloorVec = []
    Window_size = math.ceil(Fs_SPI/50)
    timeVDC_SlowVec = []
    FilteredSignalLPInitalize_Vec = []
    # for index in range(int(Window_size/2), int(Window_size*1.2) ,int(Window_size*0.7)):
    #     ind = index
    #     Vdc_Window = Vdc_fast_With_arc[ind:ind + Window_size]
    #     FilteredSignalLPInitalize = butter_lowpass_filter1(Vdc_Window, fc, Fs_SPI, order=orderFilter)  # low pass
    #     FilteredSignalLPInitalize_Vec.extend(FilteredSignalLPInitalize)

    # FirstValMin = min(FilteredSignalLPInitalize_Vec)
    # FirstIndxMin = FilteredSignalLPInitalize_Vec.index(FirstValMin)
    #
    for index in range(0, int(Window_size*2) ,int(Window_size)):
        ind = index
        Vdc_Window = Vdc_fast_With_arc[ind:ind + Window_size]
        FilteredSignalLPInitalize = butter_lowpass_filter1(Vdc_Window, fc, Fs_SPI, order=orderFilter)  # low pass
        FilteredSignalLPInitalize_Vec.extend(FilteredSignalLPInitalize)

    #StartM = FilteredSignalLPInitalize_Vec[:Window_sizeSync]
    FilteredSignalLPInitalize_VecCUT = FilteredSignalLPInitalize_Vec[int(Window_size / 2):int(Window_size*1.2)]
    FirstValMin = min(FilteredSignalLPInitalize_VecCUT)
    FirstIndxMin = FilteredSignalLPInitalize_VecCUT.index(FirstValMin) + int(Window_size / 2)

    ValVdc_WideWindowMinVec = [FirstValMin]
    IndxVdc_WideWindowMinVec = [int(FirstIndxMin)]
    FilteredSignalLP_Vec = []
    time_FilteredSignalLP_Vec = time_spi_fast[FirstIndxMin:]
    n = 1
    # Extract locals minimums per cycle
    for index in range( int(FirstIndxMin), len(Vdc_fast_With_arc),int(Window_size)):
        timeSlow = time_spi_fast[index]
        timeVDC_SlowVec.append(timeSlow)
        ind = index
        Vdc_WideWindow = Vdc_fast_With_arc[ind:ind + Window_size]
        Vdc_WideWindowFiltered = butter_lowpass_filter1(Vdc_WideWindow, fc, Fs_SPI, order=orderFilter)  # low pass
        # ValVdc_WideWindowMin = min(Vdc_WideWindowFiltered)
        Vdc_WideWindowFilteredList = Vdc_WideWindowFiltered.tolist()
        try:
            # IndxVdc_WideWindowMin = Vdc_WideWindowFilteredList.index(min(Vdc_WideWindowFilteredList[35:])) + 35
            # ValVdc_WideWindowMin = min(Vdc_WideWindowFilteredList[35:])
            ValVdc_WideWindowMin = min(Vdc_WideWindowFilteredList[35:])
            IndxVdc_WideWindowMin = Vdc_WideWindowFilteredList.index(ValVdc_WideWindowMin) #+ 35
        except:
            ValVdc_WideWindowMin = min(Vdc_WideWindowFilteredList)
            IndxVdc_WideWindowMin = Vdc_WideWindowFilteredList.index(ValVdc_WideWindowMin)
            print("Problem " + Name)
        # ValVdc_WideWindowMin = Vdc_WideWindowFilteredList[IndxVdc_WideWindowMin]
        ValVdc_WideWindowMinVec.append(ValVdc_WideWindowMin)
        IndxVdc_WideWindowMinVec.append(index +  int(IndxVdc_WideWindowMin))
        # IndxVdc_WideWindowMinVec.append(int(FirstIndxMin) + int(IndxVdc_WideWindowMin + int(n *Window_size)))
        n = n+1
        # FilteredSignalLP_Vec.extend(Vdc_WideWindowFiltered)
        FilteredSignalLP_Vec.extend(Vdc_WideWindowFilteredList)

    return FilteredSignalLP_Vec ,time_FilteredSignalLP_Vec , ValVdc_WideWindowMinVec, IndxVdc_WideWindowMinVec

def VDC_Drop_calc(VDC_Slow,timeS, windowSize = 20, filter_size=15, samTH = 12):
    buffer_size = 50
    min_in_filterEachbuffer = []
    VecThresh = [0 for x in range(buffer_size - windowSize - filter_size )]
    #VecThresh = [0 for x in range(buffer_size)]
    timeWindVec = []
    for index in range(buffer_size, len(VDC_Slow)):
        timeWind = timeS[index]
        timeWindVec.append(timeWind)
        ind = index - windowSize - filter_size
        buffer1 = VDC_Slow[ind:ind + buffer_size]
        window1 = buffer1[-windowSize:]
        filter1 = buffer1[-(windowSize+filter_size): -windowSize]
        # buffer1 = VDC_Slow[ind:ind + buffer_size]
        # filter1 = buffer1[:filter_size]
        # window1 = buffer1[filter_size:filter_size + windowSize]
        min_in_filter = np.mean(filter1)
        min_in_filterEachbuffer += [min_in_filter]
        SortWind = numpy.sort(window1)
        SortWindFlip = SortWind # np.flip(SortWind)
        if len(SortWindFlip) > samTH:
            K_sampValWin = SortWindFlip[samTH-1]
            VecThresh += [K_sampValWin - min_in_filter]
        if (len(SortWindFlip) <= samTH) & (SortWindFlip.size > 0):
            K_sampValWin = SortWindFlip[0]
            VecThresh += [K_sampValWin - min_in_filter]
        index += 1
    TempVEC = VecThresh
    # if (len(TempVEC)) < len(VDC_Slow):
    #     for i in range(len(VDC_Slow) - len(TempVEC)):
    #         VecThresh.append(K_sampValWin - min_in_filter)
    VecThresh = [0 if x > 0 else float(x) for x in VecThresh]
    return np.abs(VecThresh),timeWindVec

def Vdc_Alg(Vdc_fast_With_arc,Fs_SPI,time_spi_fast,Name,orderFilter ,fc ):
    Window_size = int(math.ceil(Fs_SPI/50))
    timeVDC_SlowVec = []
    FilteredSignalLPInitalize_Vec = []
    for index in range(0, int(Window_size*2) ,int(Window_size)):
        ind = index
        Vdc_Window = Vdc_fast_With_arc[ind:ind + Window_size]
        FilteredSignalLPInitalize = butter_lowpass_filter1(Vdc_Window, fc, Fs_SPI, order=orderFilter)  # low pass
        FilteredSignalLPInitalize_Vec.extend(FilteredSignalLPInitalize)

    FilteredSignalLPInitalize_VecCUT = FilteredSignalLPInitalize_Vec[int(Window_size / 2):int(Window_size*1.2)]
    FirstValMin = min(FilteredSignalLPInitalize_VecCUT)
    FirstIndxMin = FilteredSignalLPInitalize_VecCUT.index(FirstValMin) + int(Window_size / 2)
    ValVdc_WideWindowMinVec = []
    ValVdc_WideWindowMinVec2 = []
    # IndxVdc_WideWindowMinVec = [int(FirstIndxMin)]
    IndxVdc_WideWindowMinVec = []
    FilteredSignalLP_Vec = []
    time_FilteredSignalLP_Vec = time_spi_fast[FirstIndxMin:]
    ExccedTHCounterVecF =[]
    LowerThVec = []
    n = 0
    k = 0
    Cyc_n = 0
    diffB = 0
    diffBVec = []
    LowerThIndxVec = []
    ExccedTHCounterVecFIndx = []
    diffBVecIndx = []
    VdcMeanVec = []
    VdcMeanVecIndx = []
    # Extract locals minimums per cycle
    # for index in range( int(FirstIndxMin), len(Vdc_fast_With_arc),int(Window_size)):
    for index in range( int(FirstIndxMin), len(Vdc_fast_With_arc),int(Window_size)):
        timeSlow = time_spi_fast[index]
        timeVDC_SlowVec.append(timeSlow)
        ind = index
        k = k +1
        Cyc_n = Cyc_n + 1
        Vdc_WideWindow = Vdc_fast_With_arc[ind:ind + Window_size]

        try:
            Vdc_WideWindowFiltered = butter_lowpass_filter1(Vdc_WideWindow, fc, Fs_SPI, order=orderFilter)  # low pass

        except:
            break
        # VdcAvg = np.mean(Vdc_WideWindow)
        # VdcMeanVec.extend([VdcAvg])
        # VdcMeanVecIndx.extend([index])
        # if len(VdcMeanVec) >=2:
        #     VdcAvg = np.mean(Vdc_WideWindow)
        #     VdcMeanVec.extend([VdcAvg])
        #     VdcMeanVecIndx.extend([index])

        Vdc_WideWindowFilteredList = Vdc_WideWindowFiltered.tolist()
        FilteredSignalLP_Vec.extend(Vdc_WideWindowFilteredList)
        if index == int(FirstIndxMin):
            # ValVdc_WideWindowMinVec2 = [Vdc_WideWindowFilteredList[0]]
            # ValVdc_WideWindowMinVec = [Vdc_WideWindowFilteredList[0]]
            ValVdc_WideWindowMin = Vdc_WideWindowFilteredList[0]
            #IndxVdc_WideWindowMin = Vdc_WideWindowFilteredList.index(ValVdc_WideWindowMin)
            IndxVdc_WideWindowMinVec.append(index)
            # int(FirstIndxMin)

        # try:
        #     #ValVdc_WideWindowMin = min(Vdc_WideWindowFilteredList[int(Window_size/2+3):])
        #     ValVdc_WideWindowMin = Vdc_WideWindowFilteredList[-1]
        # except:
        #     #ValVdc_WideWindowMin = min(Vdc_WideWindowFilteredList)
        #     print("Problem " + Name)
        # if len(FilteredSignalLP_Vec) == Window_size:
        #     ValVdc_WideWindowMin = Vdc_WideWindowFilteredList[-1]
        #     IndxVdc_WideWindowMin = Vdc_WideWindowFilteredList.index(ValVdc_WideWindowMin)
        #     IndxVdc_WideWindowMinVec.append(index + int(IndxVdc_WideWindowMin))
        if len(FilteredSignalLP_Vec) > Window_size:
            Vdc_WideWindowFilteredList1past = FilteredSignalLP_Vec[-int(2*Window_size):-Window_size]
            ValVdc_WideWindowMin11 = min(Vdc_WideWindowFilteredList1past[int(4*Window_size/5):])
            ValVdc_WideWindowMin22 = min(Vdc_WideWindowFilteredList[:int(Window_size/4)])
            if ValVdc_WideWindowMin22<ValVdc_WideWindowMin11:
                ValVdc_WideWindowMin = ValVdc_WideWindowMin22
                IndxVdc_WideWindowMin = Vdc_WideWindowFilteredList.index(ValVdc_WideWindowMin)
                IndxVdc_WideWindowMinVec.append(index + int(IndxVdc_WideWindowMin))
            else:
                ValVdc_WideWindowMin = ValVdc_WideWindowMin11
                IndxVdc_WideWindowMin = Vdc_WideWindowFilteredList1past.index(ValVdc_WideWindowMin)
                IndxVdc_WideWindowMinVec.append(index - Window_size + int(IndxVdc_WideWindowMin))

        # if n==1: check how closed the indexs
        #     ValVdc_WideWindowMin22 = min(Vdc_WideWindowFilteredList[:int(Window_size / 2 )])
        # if ValVdc_WideWindowMin22<ValVdc_WideWindowMin11:
        #     ValVdc_WideWindowMin = ValVdc_WideWindowMin22
        # else:
        #     ValVdc_WideWindowMin = ValVdc_WideWindowMin11

        #IndxVdc_WideWindowMin = Vdc_WideWindowFilteredList.index(ValVdc_WideWindowMin)
        #IndxVdc_WideWindowMinVec.append(index +  int(IndxVdc_WideWindowMin))
        ValVdc_WideWindowMinVec.append(ValVdc_WideWindowMin)
        ValVdc_WideWindowMinVec2.append(ValVdc_WideWindowMin)

        ExccedTHCounterF = 0
        if k <= 4:
            if k == 3:
                Refference_Mins = np.array(ValVdc_WideWindowMinVec2[-3:])
                Referrence_MeanofMins = np.mean(Refference_Mins)
                Referrence_STDofMins = np.std(Refference_Mins)
                # LowerTh = Referrence_MeanofMins*0.99975# - 0.1 * Referrence_STDofMins
                LowerTh = Referrence_MeanofMins*0.99978 #- 0.1 * Referrence_STDofMins
                LowerThVec.extend([LowerTh]*Window_size )
                LowerThIndxVec.extend(list(range(int(index+Window_size), int(index+2*Window_size))))
            if k == 4:
                checkCycle = FilteredSignalLP_Vec[int(-Window_size):]
                for value in checkCycle:
                    if value < LowerTh:
                        ExccedTHCounterF += 1
                if ExccedTHCounterF >= 270 :  # int(ExccedTHCounterF / 2)):
                    ExccedTHCounterVecF.extend([ExccedTHCounterF])
                    ExccedTHCounterVecFIndx.extend([index+Window_size])
                else:
                    ExccedTHCounterVecF.extend([0])
                    ExccedTHCounterVecFIndx.extend([index+Window_size])

        if k > 4 :
            # LowerTh = np.mean(np.array(ValVdc_WideWindowMinVec2[-6:-2]))*0.99975# - 0.1*np.std(np.array(ValVdc_WideWindowMinVec2[-7:-3]))
            LowerTh = np.mean(np.array(ValVdc_WideWindowMinVec2[-5:-1]))*0.99978 #-0.1*np.std(np.array(ValVdc_WideWindowMinVec2[-6:-2]))
            LowerThVec.extend([LowerTh]*Window_size)
            LowerThIndxVec.extend(list(range(int(index ), int(index + Window_size))))
            checkCycle = FilteredSignalLP_Vec[int(-Window_size):]
            for value in checkCycle:
                if value < LowerTh:
                    ExccedTHCounterF += 1
            # if len(LowerThVec) >=5:
            #     diffB = LowerTh - LowerThVec[-5]
                # diffBVecIndx.extend([index])
            # else:
            #     diffB =  LowerTh - LowerThVec[-1]
            if ExccedTHCounterF >= 1:  # int(ExccedTHCounterF / 2)):
                ExccedTHCounterVecF.extend([ExccedTHCounterF])
                ExccedTHCounterVecFIndx.extend([index+Window_size])
            else:
                ExccedTHCounterVecF.extend([0])
                ExccedTHCounterVecFIndx.extend([index+Window_size])
            if len(LowerThVec) >= int(5*Window_size):
                diffB = LowerTh - (LowerThVec[-int(5*Window_size)]) #np.mean(LowerThVec[-int(5*Window_size):])
                if diffB <0 :  # int(ExccedTHCounterF / 2)):
                    diffBVec.extend([-diffB])
                    diffBVecIndx.extend([index])
                else:
                    diffBVec.extend([0])
                    diffBVecIndx.extend([index])

    return FilteredSignalLP_Vec ,time_FilteredSignalLP_Vec , ValVdc_WideWindowMinVec, IndxVdc_WideWindowMinVec , ExccedTHCounterVecF , ExccedTHCounterVecFIndx,LowerThVec,LowerThIndxVec,timeVDC_SlowVec ,diffBVec,diffBVecIndx,VdcMeanVec,VdcMeanVecIndx

def energy_rise_calc(log_energy, timeEnergy, windowSize=20, filter_size=15, samTH=12, buffer_size=50):
    # VecThresh = [0 for x in range(buffer_size)]
    VecThresh = [0 for x in range(buffer_size - windowSize - filter_size)]
    timeWindVec = []
    for index in range(buffer_size, len(log_energy)):
        timeWind = timeEnergy[index]
        timeWindVec.append(timeWind)
        ind = index - windowSize - filter_size
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

def stage1_energy_calc(SignalAfterDownsample ,TimeAfterDownsample,FsAfterDownsample, fIf = 6000,alpha = 0.2857,TH=12,windowSize=20,filter_size=15,samTH=12):
    ARC_MAX_BUFFER_SIZE = 50
    Signal = np.array(SignalAfterDownsample, dtype=float)
    timeDow = np.array(TimeAfterDownsample, dtype=float)
    DClevel = 0#8200
    DcLevelCounter = 0
    DCLevelSum = 0
    PlaceInBit = 0
    arcbw = 35
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
        if DcLevelCounter == SamplesPerBitF:
            DClevel = DCLevelSum / DcLevelCounter
            DCLevelSum = 0
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
    log_energy1 = 10 * np.log10()
    log_energy2 = np.insert(log_energy1, 0, 0)
    log_energy = log_energy2[2:]
    # timeEnergy = timeEnergy[0:len(log_energy)]
    timeEnergy2 = np.arange(0, timeDow[-1] + 1 / arcbw, 1 / arcbw)
    timeEnergy = timeEnergy2[2:]
    return log_energy,timeEnergy



def Vdc(Vdc_fast_With_arc,Fs_SPI,time_spi_fast,Name,orderFilter ,fc ):
    Window_size = int(math.ceil(Fs_SPI/50))
    timeVDC_SlowVec = []
    FilteredSignalLPInitalize_Vec = []
    for index in range(0, int(Window_size*2) ,int(Window_size)):
        ind = index
        Vdc_Window = Vdc_fast_With_arc[ind:ind + Window_size]
        FilteredSignalLPInitalize = butter_lowpass_filter1(Vdc_Window, fc, Fs_SPI, order=orderFilter)  # low pass
        FilteredSignalLPInitalize_Vec.extend(FilteredSignalLPInitalize)

    FilteredSignalLPInitalize_VecCUT = FilteredSignalLPInitalize_Vec[int(Window_size / 2):int(Window_size*1.2)]
    FirstValMin = min(FilteredSignalLPInitalize_VecCUT)
    FirstIndxMin = FilteredSignalLPInitalize_VecCUT.index(FirstValMin) + int(Window_size / 2)
    ValVdc_WideWindowMinVec2 = []
    IndxVdc_WideWindowMinVec = []
    FilteredSignalLP_Vec = []
    time_FilteredSignalLP_Vec = time_spi_fast[FirstIndxMin:]
    ExccedTHCounterVecF =[]
    LowerThVec = []
    n = 0
    k = 0
    Cyc_n = 0
    diffB = 0
    diffBVec = []
    LowerThIndxVec = []
    ExccedTHCounterVecFIndx = []
    diffBVecIndx = []
    VdcMeanVec = []
    VdcMeanVecIndx = []
    VdcAvgdiffVec = []
    VdcAvgdiffVecIndx = []
    MultiplyVec = []
    ValVdc_WideWindowMinB = 0
    ValVdc_WideWindowMinE = 0
    LowerVec = []
    LowerVecIndx = []
    # Extract locals minimums per cycle
    # for index in range( int(FirstIndxMin), len(Vdc_fast_With_arc),int(Window_size)):
    for Index in range( int(FirstIndxMin), len(Vdc_fast_With_arc),int(Window_size)):
        timeSlow = time_spi_fast[Index]
        timeVDC_SlowVec.append(timeSlow)
        ind = Index
        k = k +1
        Cyc_n = Cyc_n + 1

        Vdc_WideWindow = Vdc_fast_With_arc[ind:ind + Window_size]

        try:
            Vdc_WideWindowFiltered = butter_lowpass_filter1(Vdc_WideWindow, fc, Fs_SPI, order=orderFilter)  # low pass

        except:
            break

        Vdc_WideWindowFilteredList = Vdc_WideWindowFiltered.tolist()
        FilteredSignalLP_Vec.extend(Vdc_WideWindowFilteredList)
        if Index == int(FirstIndxMin):
            ValVdc_WideWindowMinB = Vdc_WideWindowFilteredList[0]
            ValVdc_WideWindowMinE = min(Vdc_WideWindowFilteredList[int(Window_size*0.8):])
            ValVdc_WideWindowMinEIndx = Vdc_WideWindowFilteredList.index(ValVdc_WideWindowMinE)+ Index
            ValVdc_WideWindowMinBIndx = Vdc_WideWindowFilteredList.index(ValVdc_WideWindowMinB)+ Index

        else:
            ValVdc_WideWindowMinB = min(Vdc_WideWindowFilteredList[:int(Window_size*0.29)])#,ValVdc_WideWindowMinE])
            ValVdc_WideWindowMinBIndx = Vdc_WideWindowFilteredList.index(ValVdc_WideWindowMinB) + Index

            if ValVdc_WideWindowMinB < ValVdc_WideWindowMinE:
                ValVdc_WideWindowMinBIndx = ValVdc_WideWindowMinBIndx
                ValVdc_WideWindowMinB = ValVdc_WideWindowMinB
            else:
                ValVdc_WideWindowMinBIndx = ValVdc_WideWindowMinEIndx
                ValVdc_WideWindowMinB = ValVdc_WideWindowMinE
            try:
                ValVdc_WideWindowMinE = min(Vdc_WideWindowFilteredList[int(Window_size*0.7):])#min([Vdc_WideWindowFilteredList[:int(Window_size/2-15)],ValVdc_WideWindowMinE])
                ValVdc_WideWindowMinEIndx = Vdc_WideWindowFilteredList.index(ValVdc_WideWindowMinE)+ Index
            except:
                ValVdc_WideWindowMinE = Vdc_WideWindowFilteredList[-1]#min(Vdc_WideWindowFilteredList[int(Window_size/2):])#min([Vdc_WideWindowFilteredList[:int(Window_size/2-15)],ValVdc_WideWindowMinE])
                ValVdc_WideWindowMinEIndx = Vdc_WideWindowFilteredList.index(ValVdc_WideWindowMinE)+ Index
        ValVdc_WideWindowMinVec2.append(ValVdc_WideWindowMinB)
        IndxVdc_WideWindowMinVec.append(ValVdc_WideWindowMinBIndx)


        ExccedTHCounterF = 0

        VdcAvg = np.mean(Vdc_WideWindow)
        VdcMeanVec.extend([VdcAvg])
        VdcMeanVecIndx.extend([Index+Window_size])

        if len(ValVdc_WideWindowMinVec2) >= 5:
            ###
            VdcAvgdiff1 = (1 - VdcMeanVec[-2] / VdcMeanVec[-3]) * 400
            VdcAvgdiff2 = (1 - VdcMeanVec[-2] / VdcMeanVec[-4]) * 600
            VdcAvgdiff3 = (1 - VdcMeanVec[-2] / VdcMeanVec[-5]) * 900

            VdcAvgdiff11 = float(0 if VdcAvgdiff1 < 0 else VdcAvgdiff1)
            VdcAvgdiff22 = float(0 if VdcAvgdiff2 < 0 else VdcAvgdiff2)
            VdcAvgdiff33 = float(0 if VdcAvgdiff3 < 0 else VdcAvgdiff3)

            VdcAvgdiff = ((VdcAvgdiff11 + VdcAvgdiff33)) * ((VdcAvgdiff33 + VdcAvgdiff22))
            VdcAvgdiffVecIndx.extend([Index + Window_size])


            # LowerThC = np.mean(np.array(ValVdc_WideWindowMinVec2[-4:])) * 0.99976
            Lower1 = (1-  ValVdc_WideWindowMinVec2[-1] / ValVdc_WideWindowMinVec2[-2])*300
            Lower2 =  (1 - ValVdc_WideWindowMinVec2[-1] / ValVdc_WideWindowMinVec2[-3])*600
            Lower3 =  (1 - ValVdc_WideWindowMinVec2[-1] / ValVdc_WideWindowMinVec2[-4])*900

            Lower11 = float(0 if Lower1 < 0 else Lower1)
            Lower22 = float(0 if Lower2 < 0 else Lower2)
            Lower33 = float(0 if Lower3 < 0 else Lower3)
            Lower = Lower33**2 +Lower11**2 +Lower22**2

            LowerVec.append(Lower)
            LowerVecIndx.extend([int(Index + Window_size)])


            # LowerThC = np.mean(np.array(ValVdc_WideWindowMinVec2[-4:])) * 0.99976
            LowerThC = np.mean(np.array(ValVdc_WideWindowMinVec2[-5:-2])) * 0.9997
            checkCycle = FilteredSignalLP_Vec[int(-2*Window_size):int(-Window_size)]#
            LowerThVec.extend([LowerThC]*Window_size)
            LowerThIndxVec.extend(list(range(int(Index - Window_size ), int(Index ))))

            for value in checkCycle:
                if value < LowerThC:
                    ExccedTHCounterF += 1
            ExccedTHCounterVecFIndx.extend([Index + Window_size])

            # diffB = -((ValVdc_WideWindowMinVec2[-2]*(1/3)+ValVdc_WideWindowMinVec2[-1]*(2/3)) - np.mean(ValVdc_WideWindowMinVec2[-5:-2])) #np.mean(LowerThVec[-int(5*Window_size):])
            diffB = -((ValVdc_WideWindowMinVec2[-2]*(1/2)+ValVdc_WideWindowMinVec2[-1]*(1/2)) - np.mean(ValVdc_WideWindowMinVec2[-5:-2])) #np.mean(LowerThVec[-int(5*Window_size):])
            if ExccedTHCounterF >= 1:  #
                ExccedTHCounterVecF.extend([ExccedTHCounterF/(Window_size/2)])
            else:
                ExccedTHCounterVecF.extend([0])

            if VdcAvgdiff > 0:
                VdcAvgdiffVec.extend([VdcAvgdiff])
            else:
                VdcAvgdiff = 0
                VdcAvgdiffVec.extend([0])

            if diffB > 0 :
                diffBVec.extend([diffB])
                diffBVecIndx.extend([Index+Window_size])
                ExccedTHCounter = ExccedTHCounterF / (Window_size / 2)
                Multiply = diffB * ExccedTHCounter * VdcAvgdiff
                MultiplyVec.extend([Multiply])

            else:
                diffBVec.extend([0])
                diffBVecIndx.extend([Index+Window_size])
                MultiplyVec.extend([0])


    return FilteredSignalLP_Vec ,time_FilteredSignalLP_Vec , ValVdc_WideWindowMinVec2, IndxVdc_WideWindowMinVec , ExccedTHCounterVecF , ExccedTHCounterVecFIndx,LowerThVec,LowerThIndxVec ,diffBVec,diffBVecIndx ,VdcAvgdiffVec,VdcAvgdiffVecIndx,MultiplyVec,LowerVec,LowerVecIndx



def IacProcess(Iac,FsAfterDownsample):
    GridFS = 50
    Mean_Size = int(FsAfterDownsample / GridFS)
    RmsIacVEC = []
    for index in range(0, len(Iac) - Mean_Size, Mean_Size):
        bufferIac = Iac[index:index + Mean_Size]
        RmsIac = np.sqrt(np.mean(bufferIac ** 2))
        RmsIacVEC.append(RmsIac)
    Fs = 50
    T = 1 / Fs
    N = len(RmsIacVEC)
    time = np.linspace(0, N * T, N)

    return RmsIacVEC, time

def Counter_VDC_Energy_calc(log_energy,VDC_Slow,timeEnergy, windowSize = 20, filter_size=15, TH_Energy = 12,TH_Vdc = 0.01):
    buffer_size = 50
    EnergyCounterVector = [0 for x in range(buffer_size - windowSize - filter_size )]
    VdcCounterVector =  [0 for x in range(buffer_size - windowSize - filter_size )]

    timeCounterVec = []
    for index in range(buffer_size, len(log_energy)):
        EnergyCounter = 0
        VdcCounter = 0
        ind = index - windowSize - filter_size
        timeCounter = timeEnergy[index]
        timeCounterVec.append(timeCounter)
        # Energy Fill buff
        bufferE = log_energy[ind:ind + buffer_size]
        windowE = bufferE[-windowSize:]
        filterE = bufferE[-(windowSize+filter_size): -windowSize]
        # VDC_Slow Fill buff
        bufferV = VDC_Slow[ind:ind + buffer_size]
        windowV = bufferV[-windowSize:]
        filterV = bufferV[-(windowSize+filter_size): -windowSize]

        min_filterEnergy = np.min(filterE)
        AVG_in_filter_V = np.mean(filterV)
        for value in windowE:
            if value >= (min_filterEnergy+ TH_Energy):
                EnergyCounter+=1
        EnergyCounterVector +=[EnergyCounter]

        for value in windowV:
            if value < (AVG_in_filter_V- TH_Vdc):
                VdcCounter+=1
        VdcCounterVector +=[VdcCounter]
    return EnergyCounterVector,VdcCounterVector,timeCounterVec
def first_gt_index(lst, k):
    # creating a heap from the list
    heap = list(lst)
    heapq.heapify(heap)

    # using heappop() to find index of first element
    # just greater than k
    #print(heap)
    for i, val in enumerate(heap):
        if val >= k:
            res = i
            break
    else:
        res = None

    return res

def extendFeatures(FirstCOND_Vec, SecondCOND_Vec, ThirdCOND_Vec,FourthCOND_Vec,ClassifyVec):
    ExccedTHCounterVecF_VecEXtend.extend(FirstCOND_Vec)
    diffBVec_VecEXtend.extend(SecondCOND_Vec)
    VdcAvgdiffVec_VecEXtend.extend(ThirdCOND_Vec)
    LowerVec_VecEXtend.extend(FourthCOND_Vec)
    ClassifyVecEXtend.extend(ClassifyVec)
    return ExccedTHCounterVecF_VecEXtend,diffBVec_VecEXtend,VdcAvgdiffVec_VecEXtend,LowerVec_VecEXtend,ClassifyVecEXtend

def labeling(RX_RAWC, time_RX_RAWC,log_energy,timeEnergy,diffBVec):#,VdcSlow,VecThreshV,VdcCounterVector):
    Classify = np.zeros(len(diffBVec))
    flag = 0
    if np.max(np.abs(RX_RAWC)) >= 295 :
        RX_RAWClist = [-x if x < 0 else float(x) for x in RX_RAWC] #RX_RAWC#.tolist()             Lower11 = float(0 if Lower1 < 0 else Lower1)
        IndMax =  [ n for n,i in enumerate(RX_RAWClist) if i>=295 ][0] # RX_RAWClist.index(max(RX_RAWClist))

        t_Max = time_RX_RAWC[IndMax] #IndMax/16667 # time_RX_RAWC[IndMax]
        Indx_end = int(t_Max * 50) - 5
        if Indx_end<50:
            flag = 2
            Classify = np.zeros(len(diffBVec))
        else:
            S = np.min([Indx_end + 10,len(Classify)])
            Classify[Indx_end-10:S] = 4#1
            flag = 1
    if flag == 2 or flag == 0:
        timeEndRecE = timeEnergy[-1]
        IndxStartSearcRecE = int((timeEndRecE - 1.5)*35)
        log_energylist = log_energy.tolist()
        IND_E_start = log_energylist.index(max(log_energylist[IndxStartSearcRecE:]))-8
        IND_c_start = int(IND_E_start*50/35)
        Classify[IND_c_start:-3] = 1  # 1
    return Classify

def cut_indxS(RX_RAW,Vdc_fast1,Fs_SPI):
    indicesR = np.where(np.array(RX_RAW) > 820)[0].tolist()
    if len(indicesR) == 0:
        indicesR = len(RX_RAW)
    else:
        indicesR = np.min([indicesR[0] + 3500, len(RX_RAW)])  # indicesR[0] + 2600

    indicesDiffR = np.where(np.abs(np.diff(np.array(RX_RAW))) > 200)[0].tolist()
    if len(indicesDiffR) == 0:
        indicesDiffR = len(RX_RAW)
    else:
        indicesDiffR = np.min([indicesDiffR[0] + 3500, len(RX_RAW)])

    indices = np.where(np.array(Vdc_fast1) > 765)[0].tolist()
    if len(indices) == 0:
        indices = len(Vdc_fast1)
    else:
        indices = indices[0]

    indices2 = np.where(np.array(Vdc_fast1) < 720)[0].tolist()
    if len(indices2) == 0:
        indices2 = len(Vdc_fast1)
    else:
        indices2 = indices2[0]
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

    indicesStop = np.min([indicesR, indices, indices2, indicesDiffR])
    return indicesStop

folder=r"M:\Users\LinoyL\Vdc_Criteria_new\MOHAMD_ARCS\RAW_DATA\S1"


UserName = "S1"
NumRec = 1000







Fs_SPI = 16667
Fs_Energy = 35
T_SPI = 1 / Fs_SPI
listFolder = [x[0] for x in os.walk(folder)]
Col_loc = 0
ColList = [0]
NameSTRList = []
MaxTH_E_STRList = []
MaxTH_V_STRList = []
MaxTH_I_STRList = []
ExccedTHCounterVecF_VecEXtend = []
diffBVec_VecEXtend = []
VdcAvgdiffVec_VecEXtend = []
LowerVec_VecEXtend = []
ClassifyVecEXtend = []


# plots_per_pane=13
# fig = initialize_fig(row=9, col=1, plots_per_pane=plots_per_pane, shared_xaxes=True,
#                      subplot_titles=['RX RAW','Energy','SPI Filtered Wind Vdc fast_minsMarker +Lower boundry eps','Counter Results Lower boundry eps','diffBVec','Vdc DIFF MEAN','Multiply','LowerVec','Classify'])

plots_per_pane=4
fig = initialize_fig(row=4, col=1, plots_per_pane=plots_per_pane, shared_xaxes=False,
                     subplot_titles=['RX RAW SPI','Energy SPI cut','Energy SPI No_Cut','Energy PWR'])
# plots_per_pane=4
# fig = initialize_fig(row=4, col=1, plots_per_pane=plots_per_pane, shared_xaxes=False,
#                      subplot_titles=['RX RAW SPI','Energy SPI','Energy PWR cut like spi','Energy PWR'])

file_arr_list=list(range(1, NumRec))
for i in range(0,len(listFolder)):
    for k, filename in enumerate([f for f in os.listdir(listFolder[i]) if (f.endswith('.txt') and 'Rec001 spi' in f)]): #or (f.endswith('.log') and 'pwr' in f)]):
        filename1 = f"{listFolder[i]}/{filename}"
        Name = filename[:-4]
        splitFile = Name.split()
        End  = filename[-3:]
        StringFile = ' '.join(map(str, splitFile[:2]))
        StringEndFile = ' '.join(map(str, splitFile[-3:]))

        Iac_L1, Iac_L2, Iac_L3, RX_RAW, Vdc_fast1, Vac_L1 = SPI_Reading.read_file(filename1)
        indicesStop = cut_indxS(RX_RAW, Vdc_fast1, Fs_SPI)

        Vdc_fast2 = Vdc_fast1[:indicesStop]
        RX_RAWC = RX_RAW[:indicesStop]
        L = len(RX_RAWC)
        time_RX_RAWC = np.linspace(0, L * T_SPI, L)
        time_RX_RAW = np.linspace(0, len(RX_RAW) * T_SPI, len(RX_RAW)) # spi no cut

        Vdc_fast = Vdc_fast2 #[x  for x in Vdc_fast2]
        Vdc_fastRemoveDC = Vdc_fast
        N = len(Vdc_fast)
        time_spi_fast = np.linspace(0, N * T_SPI, N)
        VdcSlow, time = VdcProcess(Vdc_fast, Fs_SPI)
        VecThreshV,timeWindVec = VDC_Drop_calc(VdcSlow, time, windowSize=20, filter_size=15, samTH=12)
        log_energy, timeEnergy = stage1_energy_calc(RX_RAWC, time_RX_RAWC, Fs_SPI , fIf=6000,alpha=0.2857, TH=12, windowSize=20, filter_size=15, samTH=12)
        VecThreshE, timeTH_E = energy_rise_calc(log_energy, timeEnergy, windowSize=20, filter_size=15, samTH=12)
        EnergyCounterVector,VdcCounterVector,timeCounterVec = Counter_VDC_Energy_calc(log_energy, VdcSlow, timeEnergy, windowSize=20, filter_size=15, TH_Energy=12,TH_Vdc=0.01)
        FilteredSignalLP_Vec ,time_FilteredSignalLP_Vec , ValVdc_WideWindowMinVec, IndxVdc_WideWindowMinVec , ExccedTHCounterVecF , ExccedTHCounterVecFIndx,LowerThVec,LowerThIndxVec ,diffBVec,diffBVecIndx,VdcAvgdiffVec,VdcAvgdiffVecIndx,MultiplyVec,LowerVec,LowerVecIndx= Vdc(Vdc_fast, Fs_SPI, time_spi_fast,Name, orderFilter=1, fc=65)

        Classify = labeling(RX_RAWC, time_RX_RAWC, log_energy, timeEnergy, diffBVec)
        N_C = len(Classify)
        T_50 = 1/50
        ClassifyT = np.linspace(time_spi_fast[diffBVecIndx[0]], N_C * T_50 +1/50, N_C)

        Time_mins = time_spi_fast[IndxVdc_WideWindowMinVec]
        timeLowerThVec = time_spi_fast[LowerThIndxVec]
        timeExccedTHCounterVecF = time_spi_fast[ExccedTHCounterVecFIndx[:-2]]
        timediffBVec = time_spi_fast[diffBVecIndx[:-1]]
        VdcAvgdiffVecTIME = time_spi_fast[VdcAvgdiffVecIndx[:-1]]
        LowerVecTime = time_spi_fast[LowerVecIndx[:-1]]

        F1 = ExccedTHCounterVecF
        F2 = diffBVec
        F3 = VdcAvgdiffVec
        F4 = LowerVec
        LABEL = Classify # I need to detect wo is the shorter or the end of 1 in classify - i need all of them the same lengt and ther logistic regression
        log_energyNoCut, timeEnergyNoCut = stage1_energy_calc(RX_RAW, time_RX_RAW, Fs_SPI , fIf=6000,alpha=0.2857, TH=12, windowSize=20, filter_size=15, samTH=12)

        ExccedTHCounterVecF_VecEXtend, diffBVec_VecEXtend, VdcAvgdiffVec_VecEXtend, LowerVec_VecEXtend, ClassifyVecEXtend = extendFeatures(ExccedTHCounterVecF, diffBVec, VdcAvgdiffVec, LowerVec,Classify)
        for kPWR, filenamePWR in enumerate(
                [fPWR for fPWR in os.listdir(listFolder[i]) if fPWR.endswith('.log') and StringFile + ' pwr ' + StringEndFile in fPWR]):
            filenamePWR = f"{listFolder[i]}/{filenamePWR}"
            NamePWR = filenamePWR[:-4]
            # EXTRACT Feature from pwr log file
            energiesPWR, EnergyCounterVecPWR,IacCounterVecPWR, VdcVecPWR,IacVecPWR = Jup_pwr_vdc_criteria.read_log_file(filenamePWR)
            N_energiesPWR = len(energiesPWR)
            T_energiesPWR = 1 / 35
            timeenergiesPWR = np.linspace(0, N_energiesPWR * T_energiesPWR, N_energiesPWR)

        diffTimePWR_SPI = timeenergiesPWR[-1] - timeEnergyNoCut[-1]##########

        fig.add_trace(go.Scattergl(x=time_RX_RAWC[:],
                                   y=RX_RAWC[:],
                                   name='RX_RAW' + 'Rec' + StringFile , mode="lines",
                                   visible=False,
                                   showlegend=True),
                      row=1, col=1)

        fig.add_trace(go.Scattergl(x=timeEnergy[:],
                                   y=log_energy[:],
                                   name='Energy SPI cut' + 'Rec' + StringFile , mode="lines",
                                   visible=False,
                                   showlegend=True),
                      row=2, col=1)
        fig.add_trace(go.Scattergl(x=timeEnergyNoCut[:],
                                   y=log_energyNoCut[:],
                                   name='Energy SPI No_Cut' + 'Rec' + StringFile , mode="lines",
                                   visible=False,
                                   showlegend=True),
                      row=3, col=1)
        fig.add_trace(go.Scattergl(x=timeenergiesPWR[:],
                                   y=energiesPWR[:],
                                   name='Energy PWR' + 'Rec' + StringFile , mode="lines",
                                   visible=False,
                                   showlegend=True),
                      row=4, col=1)

        # fig.add_trace(go.Scattergl(x=time_FilteredSignalLP_Vec[:],
        #                            y=FilteredSignalLP_Vec[:],
        #                            name='Filtered Wind Vdc fast ' + 'Rec' + StringFile , mode="lines",
        #                            visible=False,
        #                            showlegend=True),
        #               row=3, col=1)
        #
        # fig.add_trace(go.Scattergl(x=Time_mins[:],
        #                            y=ValVdc_WideWindowMinVec[:],
        #                            name='Local Mins ' + 'Rec' + StringFile ,mode="markers",
        #                            visible=False,
        #                            showlegend=True),
        #               row=3, col=1)
        #
        # fig.add_trace(go.Scattergl(x=timeLowerThVec[:],
        #                            y=LowerThVec[:],
        #                            name='Lower boundry eps ' + 'Rec' + StringFile ,mode="lines",
        #                            visible=False,
        #                            showlegend=True),
        #               row=3, col=1)
        # fig.add_trace(go.Scattergl(x=timeExccedTHCounterVecF[:],#Time_mins[2:],
        #                            y=ExccedTHCounterVecF[:],
        #                            name='Counter Results Lower boundry eps ' + 'Rec' + StringFile ,mode="lines",
        #                            visible=False,
        #                            showlegend=True),
        #               row=4, col=1)
        # fig.add_trace(go.Scattergl(x=timeExccedTHCounterVecF[:],#Time_mins[2:],
        #                            y=0.7* np.ones(len(timeExccedTHCounterVecF[:])),
        #                            name='200 th counter ' + 'Rec' + StringFile ,mode="lines",
        #                            visible=False,
        #                            showlegend=True),
        #               row=4, col=1)
        # fig.add_trace(go.Scattergl(x=timediffBVec[:],#Time_mins[2:],
        #                            y=diffBVec[:],
        #                            name='diffBVec ' + 'Rec' + StringFile ,mode="lines",
        #                            visible=False,
        #                            showlegend=True),
        #               row=5, col=1)
        # fig.add_trace(go.Scattergl(x=timediffBVec[:],#Time_mins[2:],
        #                            y=0.22* np.ones(len(timediffBVec[:]))[:],
        #                            name='th 0.2 ' + 'Rec' + StringFile ,mode="lines",
        #                            visible=False,
        #                            showlegend=True),
        #               row=5, col=1)
        # fig.add_trace(go.Scattergl(x=VdcAvgdiffVecTIME[:],#Time_mins[2:],
        #                            y=VdcAvgdiffVec[:],
        #                            name='Vdc DIFF MEAN ' + 'Rec' + StringFile ,mode="lines",
        #                            visible=False,
        #                            showlegend=True),
        #               row=6, col=1)
        # fig.add_trace(go.Scattergl(x=timediffBVec[:-1],#Time_mins[2:],
        #                            y=MultiplyVec[:-1],
        #                            name='Multiply ' + 'Rec' + StringFile ,mode="lines",
        #                            visible=False,
        #                            showlegend=True),
        #               row=7, col=1)
        # fig.add_trace(go.Scattergl(x=LowerVecTime[:],#Time_mins[2:],
        #                            y=LowerVec[:],
        #                            name='LowerVec ' + 'Rec' + StringFile ,mode="lines",
        #                            visible=False,
        #                            showlegend=True),
        #               row=8, col=1)
        # fig.add_trace(go.Scattergl(x=timediffBVec[:],#Time_mins[2:],
        #                            y=Classify[:],
        #                            name='Classify ' + 'Rec' + StringFile ,mode="lines",
        #                            visible=False,
        #                            showlegend=True),
        #               row=9, col=1)



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


fig.write_html('S1_SPI&PWR.HTML', auto_open=True, config=config)


# # Logistic Regression implementation
# TrainMAT = np.asarray([ExccedTHCounterVecF_VecEXtend, diffBVec_VecEXtend,VdcAvgdiffVec_VecEXtend])
# TrainMAT1 = np.transpose(TrainMAT)
# model = LogisticRegression(solver='liblinear', random_state=0).fit(TrainMAT1,ClassifyVecEXtend)
# model.score(TrainMAT1, ClassifyVecEXtend)
# ## Confusion Matrix logistic regression
# cm = confusion_matrix(ClassifyVecEXtend, model.predict(TrainMAT1))
# fig, ax = plt.subplots(figsize=(8, 8))
# ax.imshow(cm)
# ax.grid(False)
# ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted 0s', 'Predicted 1s'))
# ax.yaxis.set(ticks=(0, 1), ticklabels=('Actual 0s', 'Actual 1s'))
# ax.set_ylim(1.5, -0.5)
# for i in range(2):
#     for j in range(2):
#         ax.text(j, i, cm[i, j], ha='center', va='center', color='red')
# plt.show()
# print(classification_report(ClassifyVecEXtend, model.predict(TrainMAT1)))