import heapq
import math
import os
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from scipy import signal

import SPI_Reading
from my_pyplot import omit_plot as _O, plot as _P, print_chrome as _PC, clear as _PP, print_lines as _PL, send_mail as _SM
import Plot_Graphs_with_Sliders as _G
import my_tools


def initialize_fig(row=4, col=1, plots_per_pane=4, shared_xaxes=True, subplot_titles=['Rx', 'Rx', 'Rx', 'Rx']):
    all_specs = np.array([[{"secondary_y": True}] for x in range((row * col))])
    all_specs_reshaped = (np.reshape(all_specs, (col, row)).T).tolist()
    fig = make_subplots(rows=row, cols=col, specs=all_specs_reshaped, shared_xaxes=shared_xaxes, subplot_titles=subplot_titles)

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


def butter_lowpass1(cutoff, fs, order):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    # b, a = signal.butter(order, normal_cutoff, btype='low', analog=True)
    return b, a


def iir_lpf(data_wind, cutoff, fs, order):
    B, A = butter_lowpass1(cutoff, fs, order=order)
    b0 = B[0]
    b1 = B[1]
    a0 = A[0]  # Almost always 1
    a1 = A[1]  # (b0+b1) -1
    x_n_1 = data_wind[0]
    Y_IIR_1 = x_n_1  # 0#750
    yVec = [Y_IIR_1]
    for index, value in enumerate(data_wind[1:]):
        x_n = data_wind[index]
        Y_IIR = (b0 * x_n + b1 * x_n_1 - (a1) * Y_IIR_1) / a0
        yVec.append(Y_IIR)
        Y_IIR_1 = Y_IIR
        x_n_1 = x_n
    return yVec


def VdcProcess(Vdc, FsAfterDownsample):
    # current method
    # FsAfterDownsample : 16667Hz
    # Vdc : VdcFast
    arcbw = 35
    Mean_Size = int(FsAfterDownsample / arcbw)
    alpha = 1e-4
    VdcVec = [Vdc[0]]
    for index in range(1, len(Vdc)):
        VdcVec.append(alpha * Vdc[index] + (1 - alpha) * VdcVec[index - 1])
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

    return VdcVecMean, time, VdcVec, time_


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
        SortWind = np.sort(window1)
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


def stage1_energy_calc(SignalAfterDownsample, TimeAfterDownsample, FsAfterDownsample, fIf=6000, alpha=0.2857, TH=12, windowSize=20, filter_size=15, samTH=12):
    ARC_MAX_BUFFER_SIZE = 50
    Signal = np.array(SignalAfterDownsample, dtype=float)
    timeDow = np.array(TimeAfterDownsample, dtype=float)
    DClevel = 0  # 8200
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

    log_energy1 = 10 * np.log10([e / 100000 for e in EnergyVec])
    log_energy2 = np.insert(log_energy1, 0, 0)
    log_energy = log_energy2[2:]
    # timeEnergy = timeEnergy[0:len(log_energy)]
    timeEnergy2 = np.arange(0, timeDow[-1] + 1 / arcbw, 1 / arcbw)
    timeEnergy = timeEnergy2[2:]
    return log_energy, timeEnergy


def Counter_VDC_Energy_calc(log_energy, VDC_Slow, timeEnergy, windowSize=20, filter_size=15, TH_Energy=12, TH_Vdc=0.01):
    buffer_size = 50
    EnergyCounterVector = [0 for x in range(buffer_size - windowSize - filter_size)]
    VdcCounterVector = [0 for x in range(buffer_size - windowSize - filter_size)]

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
        filterE = bufferE[-(windowSize + filter_size): -windowSize]
        # VDC_Slow Fill buff
        bufferV = VDC_Slow[ind:ind + buffer_size]
        windowV = bufferV[-windowSize:]
        filterV = bufferV[-(windowSize + filter_size): -windowSize]

        min_filterEnergy = np.min(filterE)
        AVG_in_filter_V = np.mean(filterV)
        for value in windowE:
            if value >= (min_filterEnergy + TH_Energy):
                EnergyCounter += 1
        EnergyCounterVector += [EnergyCounter]

        for value in windowV:
            if value < (AVG_in_filter_V - TH_Vdc):
                VdcCounter += 1
        VdcCounterVector += [VdcCounter]
    return EnergyCounterVector, VdcCounterVector, timeCounterVec


def first_gt_index(lst, k):
    # creating a heap from the list
    heap = list(lst)
    heapq.heapify(heap)
    for i, val in enumerate(heap):
        if val >= k:
            res = i
            break
    else:
        res = None

    return res


def cut_indxS(RX_RAW, Vdc_fast1, Fs_SPI):
    indicesR = np.where(np.array(RX_RAW) > 820)[0].tolist()
    if len(indicesR) == 0:
        indicesR = len(RX_RAW)
    else:
        indicesR = np.min([indicesR[0] + 3500, len(RX_RAW)])  # indicesR[0] + 260000000000000000000000000

    indicesDiffR = np.where(np.abs(np.diff(np.array(RX_RAW))) > 205)[0].tolist()
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


def VDC_Drop_calc(VDC_Slow, timeS, windowSize=20, filter_size=15, samTH=12):
    buffer_size = 50  # 50
    min_in_filterEachbuffer = []
    VecThresh = [0 for x in range(buffer_size - windowSize - filter_size)]
    timeWindVec = []
    for index in range(buffer_size, len(VDC_Slow)):
        timeWind = timeS[index]
        timeWindVec.append(timeWind)
        ind = index - windowSize - filter_size
        buffer1 = VDC_Slow[ind:ind + buffer_size]
        window1 = buffer1[-windowSize:]
        filter1 = buffer1[-(windowSize + filter_size): -windowSize]
        min_in_filter = np.mean(filter1)
        min_in_filterEachbuffer += [min_in_filter]
        SortWind = np.sort(window1)
        SortWindFlip = SortWind
        if len(SortWindFlip) > samTH:
            K_sampValWin = SortWindFlip[samTH - 1]
            VecThresh += [K_sampValWin - min_in_filter]
        if (len(SortWindFlip) <= samTH) & (SortWindFlip.size > 0):
            K_sampValWin = SortWindFlip[0]
            VecThresh += [K_sampValWin - min_in_filter]
        index += 1
    VecThresh = [0 if x > 0 else float(x) for x in VecThresh]
    return np.abs(VecThresh), timeWindVec


def newVdcProcessCounter(VdcFast, cutoff, fs, order, windowSize=10, filter_size=25, samTH=2, TH_Vdc=0.05):
    Window_size = math.ceil(Fs_SPI / 50)
    FilteredSignalLPInitalize_Vec = []
    for index in range(0, int(Window_size * 2), int(Window_size)):
        ind = index
        Vdc_Window = VdcFast[ind:ind + Window_size]
        FilteredSignalLPInitalize = iir_lpf(Vdc_Window, cutoff, Fs_SPI, order=order)  # iir_lpf(data_wind,cutoff, fs, order)
        FilteredSignalLPInitalize_Vec.extend(FilteredSignalLPInitalize)

    FilteredSignalLPInitalize_VecCUT = FilteredSignalLPInitalize_Vec[int(Window_size / 2):int(Window_size * 1.2)]
    FirstValMin = min(FilteredSignalLPInitalize_VecCUT)
    FirstIndxMin = FilteredSignalLPInitalize_VecCUT.index(FirstValMin) + int(Window_size / 2)
    arcbw = 50
    Mean_Size = int(fs / arcbw)
    B, A = butter_lowpass1(cutoff, fs, order=order)
    b0 = B[0]
    b1 = B[1]
    a0 = A[0]  # Almost always 1
    a1 = A[1]  # (b0+b1) -1
    x_n_1 = VdcFast[int(FirstIndxMin)]
    Y_IIR_1 = x_n_1
    VdcVec = [Y_IIR_1]
    for index in range(int(FirstIndxMin + 1), int(len(VdcFast) - 1)):
        x_n = VdcFast[index]
        Y_IIR = (b0 * x_n + b1 * x_n_1 - (a1) * Y_IIR_1) / a0
        VdcVec.append(Y_IIR)
        Y_IIR_1 = Y_IIR
        x_n_1 = x_n

    T = 1 / fs
    N_ = len(VdcVec)
    time_ = np.linspace((FirstIndxMin + 1) * T, N_ * T, N_)

    VdcVecMean = []
    for index in range(0, len(VdcVec) - Mean_Size, Mean_Size):
        bufferVdc = VdcVec[index:index + Mean_Size]
        MeanVdc = np.mean(bufferVdc)
        VdcVecMean.append(MeanVdc)
    Fs = arcbw
    T = 1 / Fs
    N = len(VdcVecMean)
    time = np.linspace(0, N * T, N)
    buffer_size = 50
    VdcCounterVector = [0 for x in range(buffer_size - windowSize - filter_size)]
    # TH_Vdc = 0.04
    timeCounterVecVdc = []
    for index in range(buffer_size, len(VdcVecMean)):
        VdcCounter = 0
        ind = index - windowSize - filter_size
        timeCounterVdc = time[index]
        timeCounterVecVdc.append(timeCounterVdc)
        # VDC_Slow Fill buff
        bufferV = VdcVecMean[ind:ind + buffer_size]
        windowV = bufferV[-windowSize:]
        filterV = bufferV[-(windowSize + filter_size): -windowSize]
        Min_in_filter_V = np.min(filterV)
        for value in windowV:
            if value < (Min_in_filter_V - TH_Vdc):
                VdcCounter += 1
        VdcCounterVector += [VdcCounter]
    #####################################
    buffer_size = 50
    min_in_filterEachbuffer = []
    VecThresh = [0 for x in range(buffer_size - windowSize - filter_size)]
    timeWindVec = []
    for index in range(buffer_size, len(VdcVecMean)):
        # timeWind = time[int(index+FirstIndxMin+1)] # int(FirstIndxMin+1)
        timeWind = time[index]
        timeWindVec.append(timeWind)
        ind = index - windowSize - filter_size
        buffer1 = VdcVecMean[ind:ind + buffer_size]
        window1 = buffer1[-windowSize:]
        filter1 = buffer1[-(windowSize + filter_size): -windowSize]
        min_in_filter = np.min(filter1)
        min_in_filterEachbuffer += [min_in_filter]
        SortWind = np.sort(window1)
        SortWindFlip = SortWind  # np.flip(SortWind)
        if len(SortWindFlip) > samTH:
            K_sampValWin = SortWindFlip[samTH - 1]
            VecThresh += [K_sampValWin - min_in_filter]
        if (len(SortWindFlip) <= samTH) & (SortWindFlip.size > 0):
            K_sampValWin = SortWindFlip[0]
            VecThresh += [K_sampValWin - min_in_filter]
        index += 1
    VecThresh = [0 if x > 0 else float(x) for x in VecThresh]
    # T = 1 / fs
    # N_ = len(VdcVec)
    # time_ = np.linspace(0, N_ * T, N_)
    # was VdcVecMean, time
    return VdcVec, time_, np.abs(VecThresh), timeWindVec, VdcCounterVector, timeCounterVecVdc


###   main() is here   ###
folder = r'V:\HW_Infrastructure\Analog_Team\ARC_Data\Results\Venus3\6kW-7497F876 MLCC\New 370Vdc tests 02'
arc_types = ["Good Arc", "Good Arc (current drop)", "Good Arc (dirty Arcer + current drop)", "Good Arc (dirty Arcer)", "Good Arc (very dirty Arcer)", "Multiple Arcs", "No Arc", "Short Arc", "Short Arc (very dirty Arcer)"]
stop_k = 3
jupiter_T__venus_F = False
plot_auto_open__name = [True, 'Vdc_Eddy']
indices_stop = 4200    # 3100 in New_Vdc_Criteria, or 4200 in Vdc_Eddy


NumRec = 1000
# Fs_SPI = 16667
Fs_SPI = 12500
T_SPI = 1 / Fs_SPI
listFolder = [x[0] for x in os.walk(folder)]
# listFolder = listFolder[-4:-1:2]
listFolder = [folder + '\\' + 'Arcs for Vdc 05 (29-10-2023)']
ContTH = 2
THVdc = 0.045
W_Size = 7
F_Size = 28
CutOff_fs = 2.5
plots_per_pane = 4


fig = initialize_fig(row=4, col=1, plots_per_pane=plots_per_pane, shared_xaxes=True,
                     subplot_titles=['RX RAW', 'VDC_FAST_RAW', 'Vdc (Counter = 12,TH = 0.01,35Hz) - Old algorithm ', ' Vdc (Counter = {},TH={},50Hz)- New algorithm'.format(ContTH, THVdc)])
file_arr_list = list(range(1, NumRec))
for i in range(0, len(listFolder)):
    for k, filename in enumerate([f for f in os.listdir(listFolder[i]) if f.endswith('.txt') and 'spi' in f]):
        filename1 = f"{listFolder[i]}/{filename}"
        Name = filename[:-4]
        splitFile = Name.split()
        StringFile = ' '.join(map(str, splitFile[:2]))
        if jupiter_T__venus_F:
            Iac_L1, Iac_L2, Iac_L3, RX_RAW, Vdc_fast1, Vac_L1 = SPI_Reading.read_file(filename1)
        else:
            RX_RAW, VacL1L2, ILInt, Vdc_fast1, Vcap1, VcapBrg = SPI_Reading.read_file(filename1)
        indicesStop1 = cut_indxS(RX_RAW, Vdc_fast1, Fs_SPI)
        indicesStop = indicesStop1 + indices_stop

        Vdc_fast2 = Vdc_fast1[:indicesStop]
        RX_RAWC = RX_RAW[:indicesStop]
        L = len(RX_RAWC)
        time_RX_RAWC = np.linspace(0, L * T_SPI, L)

        Vdc_fast = [x for x in Vdc_fast2]
        Vdc_fastRemoveDC = Vdc_fast
        N = len(Vdc_fast)
        time_spi_fast = np.linspace(0, N * T_SPI, N)
        VdcSlow, time, VdcFastFiltered, time_ = VdcProcess(Vdc_fast, Fs_SPI)
        VecThreshV, timeWindVec = VDC_Drop_calc(VdcSlow, time, windowSize=20, filter_size=15, samTH=12)
        log_energy, timeEnergy = stage1_energy_calc(RX_RAWC, time_RX_RAWC, Fs_SPI, fIf=6000, alpha=0.2857, TH=12, windowSize=20, filter_size=15, samTH=12)
        VecThreshE, timeTH_E = energy_rise_calc(log_energy, timeEnergy, windowSize=20, filter_size=15, samTH=12)
        EnergyCounterVector, VdcCounterVectorOld, timeCounterVecOld = Counter_VDC_Energy_calc(log_energy, VdcSlow, timeEnergy, windowSize=20, filter_size=15, TH_Energy=12, TH_Vdc=0.01)
        ## Eddy this the function you need !!! --> newVdcProcessCounter
        VdcVecFileredNew, timeNew, VecThreshV_NN, timeWindVec_NN, VdcCounterVector, timeCounterVecVdc = newVdcProcessCounter(Vdc_fast, CutOff_fs, 16667, 1, windowSize=W_Size, filter_size=F_Size, samTH=ContTH, TH_Vdc=THVdc)
        BinaryResVdcCounterVectorOld = [1 if x >= 12 else 0 for x in VdcCounterVectorOld]
        BinaryResVdcCounterVector = [1 if x >= 2 else 0 for x in VdcCounterVector]

        fig.add_trace(go.Scattergl(x=time_RX_RAWC[:],
                                   y=RX_RAWC[:],
                                   name='RX_RAW' + 'Rec' + StringFile, mode="lines",
                                   visible=False,
                                   showlegend=True),
                      row=1, col=1, secondary_y=False)
        fig.add_trace(go.Scattergl(x=time_spi_fast[:],
                                   y=Vdc_fast[:],
                                   name='VDC_RAW' + 'Rec' + StringFile, mode="lines",
                                   visible=False,
                                   showlegend=False),
                      row=2, col=1, secondary_y=False)

        fig.add_trace(go.Scattergl(x=timeCounterVecOld[:],
                                   y=BinaryResVdcCounterVectorOld[:],
                                   name='Vdc Old - (TH COUNTER = 12 for th = 0.01) ' + 'Rec' + StringFile, mode="lines",
                                   visible=False,
                                   showlegend=True),
                      row=3, col=1, secondary_y=False)

        fig.add_trace(go.Scattergl(x=timeCounterVecVdc[:],
                                   y=BinaryResVdcCounterVector[:],
                                   name='Vdc New - (TH COUNTER = {} for th = {}) '.format(ContTH, THVdc) + 'Rec' + StringFile, mode="lines",
                                   visible=False,
                                   showlegend=True),
                      row=4, col=1, secondary_y=False)
        if k == stop_k - 1:
            break

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

config = {'scrollZoom': True, 'responsive': False, 'editable': False, 'modeBarButtonsToAdd': ['drawline', 'drawopenpath', 'drawclosedpath', 'drawcircle', 'drawrect', 'eraseshape']}
fig.write_html(f'{plot_auto_open__name[1]} (ContTH={ContTH}, THVdc={THVdc}, W_Size={W_Size}, F_Size={F_Size}, CutOff_fs={CutOff_fs}).html', auto_open=plot_auto_open__name[0], config=config)
