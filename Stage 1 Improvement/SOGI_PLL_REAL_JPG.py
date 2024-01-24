import os
import sys
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from statistics import mean
from my_pyplot import omit_plot as _O, plot as _P, print_chrome as _PC, clear as _PP, print_lines as _PL, send_mail as _SM
import Plot_Graphs_with_Sliders as _G
import my_tools


# ####   Files   #### #
path_log_folder = r'V:\HW_Infrastructure\Analog_Team\ARC_Data\Results\Jupiter\Jupiter+ Improved (7E0872F4-EC)\Stage 2 Validation (27-08-2023)\Stage 2 Validation 01 - failed (21-08-2023) - Copy'
filter_rec = ['']
path_log_name_scope = ['scope rec', '.csv']
path_log_name_spi = ['spi rec', '.txt']
files_scope = sorted([f for f in os.listdir(path_log_folder) if all(s in f.lower() for s in path_log_name_scope) and any(s in f.lower() for s in filter_rec)])
files_spi = sorted([f for f in os.listdir(path_log_folder) if all(s in f.lower() for s in path_log_name_spi) and any(s in f.lower() for s in filter_rec)])

# ####   True   ###   False   #### #
output_plots_dpi = [True, 250]
adjust_scope_input_level_scope = [True, 1.03]
decimate_scope_input = [True, 3]   # from Fs = 50kHz to 16.667kHz or 12.5kHz
plot_cut_before_scope = [True, 5000, 25000]
plot_cut_after_scope = [2500, 5000, 10000, 25000]
plot_cut_before_spi = [True, 1, -2100, 17600]        # 0 = ToF, 1 = State Machine, 2 = start trim, 3 = desired length
plot_cut_after_spi = [2500, 5000, 10000, 25000]

# ####   PLL   #### #
SysCLK = 16667
MainFreq = 500
Ki = 0.03  # Old Ki = 0.045
KpCoeff = 1
correction_factor = 0.99996   # OLD = 1.00012

############### Yalla ###############
for file_index, file_scope in enumerate(files_scope):
    rec_number = file_scope[file_scope.lower().find('rec'):file_scope.lower().find('rec') + 6]
    if 'ka2' in file_scope.lower():
        MainFreq = 492.738281
    else:
        MainFreq = 511.402344

    ############### PLL Prameters ###############
    ExtZeroCrossPhiDeviation = []
    PI2 = np.pi*2
    K1_Constant = KpCoeff*PI2/SysCLK
    ADC_MAX = 6000
    cycle = 0
    SyncingPhaseCylcesCounter = 0
    ArcPhaseSyncThrPUInt32 = 3/360
    HoldoffForSync = 0
    PHI_STEPS = 32
    PARAM_ARC_STAGE2_SYNC_CYCLES = 10
    ExtPhiSumOfDeviations = 0
    PLL_ZEROCROSS_PHI = 0.75
    ExtPhi = 0
    ExtZeroCrossPhiDeviation = []
    for i in range(PHI_STEPS):
        ExtZeroCrossPhiDeviation.append(0)
    ExtPhiIndex = 0
    ArcPhaseShiftThrPUInt32 = 7
    PhaseShiftLargerThanXDegCntr = 0
    PARAM_ARC_STAGE2_PHASE_THR_CYLCES = 10
    MaxPhaseShift = 0
    AbsAmpsSum = 0
    PLL_ABSAMPSSUM_FILTER_COMP_ALPHA = 0.875
    PLL_ABSAMPSSUM_FILTER_ALPHA = 0.125         ## 1 / PLL_ABSAMPSSUM_FILTER_SIZE (8) = 0.125
    AbsAmpsIncr = 0
    ArcPllFinalAmplitudeShift = 0
    SyncedFlag = 0
    MaxPhaseShift = 0
    AbsAmpsBefore = 0
    AbsAmpsAfter = 0
    ExtPhiStepCalc = (MainFreq/SysCLK)
    ExtPhiStep = ExtPhiStepCalc
    sign = lambda x: math.copysign(4, x)


    try:
        df = pd.read_csv(f'{path_log_folder}/{file_scope}').dropna(how='all', axis='columns')
    except:
        exit(f'exit(): file "{path_log_folder}/{file_scope}" not found')
    df['PLL'] = df['PLL'] * 2**14 / 2.5 - 8200
    plot_rows = int(len(df.columns) / 1)
    for index_df, (title_df, sub_df) in enumerate(df.items()):
        if title_df == 'PLL':  # or title_df == 'CH2'
            scope_input = sub_df
    if decimate_scope_input[0]:
        scope_input = signal.decimate(scope_input, decimate_scope_input[1])
    if adjust_scope_input_level_scope[0]:
        scope_input = (scope_input - scope_input.mean()) * adjust_scope_input_level_scope[1]
    if plot_cut_before_scope[0]:
        scope_input = scope_input[plot_cut_before_scope[1]:plot_cut_before_scope[2]]


    def Sat(x, max, min):
        y = x
        if x > max:
            y = max
        if x < min:
            y = min
        return(y)


    class SOGI2:
      def __init__(self, Ki, Freq, K1_Constant, Saturation):
        self.K1_Update = Freq*K1_Constant
        self.K1 = self.K1_Update
        self.K1_Const = K1_Constant
        self.K2 = Ki
        self.X1 = 0
        self.X2 = 0
        self.Y1 = 0
        self.Y2 = 0
        self.Ki = 0
        self.SatMax = Saturation
        self.SatMin = -Saturation


    def SOGI2_Iter(SOGI, Err):
        SOGI.Y2 = SOGI.Y1;
        SOGI.X2 = SOGI.X1;
        SOGI.X1 = SOGI.Y2 * SOGI.K1 + SOGI.X2;
        SOGI.Y1 = Sat((Err * SOGI.K2 - SOGI.X1)*SOGI.K1 + SOGI.Y2,SOGI.SatMax,SOGI.SatMin);


    def PLL_SlwIterArc():
        global SOGI
        x = SOGI.X1
        y = SOGI.Y1
        Amplitude = math.sqrt(x*x+y*y)
        InvdV = 1/(Amplitude*2*np.pi)
        return(InvdV)


    ############### Print outs ###############
    y1 = []
    y2 = []
    x1 = []
    sum_of_dev = [0]
    abs_amps_sum = []
    current_phi_dev = [0]


    ############# PLL Iteration CODE #############
    def PLL_FstIterArc_NEW(sample, before_after_power_down = 2):
        global SyncingPhaseCylcesCounter
        global cycle
        global HoldoffForSync
        global ExtPhiSumOfDeviations
        global ExtPhi
        global ExtPhiIndex
        global PhaseShiftLargerThanXDegCntr
        global AbsAmpsSum
        global AbsAmpsIncr
        global ArcPllFinalAmplitudeShift
        global SyncedFlag
        global MaxPhaseShift
        global AbsAmpsBefore
        global AbsAmpsAfter
        global ExtPhiStep

        SOGI.K1 = SOGI.K1_Update
        SOGIError = sample - SOGI.Y1
        SOGI2_Iter(SOGI,SOGIError)

        if sign(SOGI.Y1) > sign(SOGI.Y2): # Detect posedge zero crossing
            # ------------------------------------Update Values After Cross----------------------------------- #
            ZerroCross   = PLL_ZEROCROSS_PHI;
            InvdV = PLL_SlwIterArc()
            Residue = InvdV*SOGI.Y1
            ZerroCross += Residue
            ExtPhiDev = ZerroCross - ExtPhi  + correction_factor # get deviation between extrapolated phi and cross point phi
            ExtPhi = ZerroCross + ExtPhiStep
            # Summarize the individual zero cross phase deviations across 32 (defined by #PHI_STEPS) number of cycles to catch the whole interference
            ExtPhiSumOfDeviations -= ExtZeroCrossPhiDeviation[ExtPhiIndex]
            ExtZeroCrossPhiDeviation[ExtPhiIndex] = ExtPhiDev
            ExtPhiSumOfDeviations += ExtZeroCrossPhiDeviation[ExtPhiIndex]
            ExtPhiIndex = (ExtPhiIndex+1)%32
            # Look for consistent deviations smaller than ArcPhaseSyncThrPUInt32 in order to declare a synced progress
            AbsExtPhiSumOfDeviations = abs(ExtPhiSumOfDeviations) # Calculate once the abs value and use it in the next lines

            # ------------------------------------Check PLL SYNC------------------------------------ #
            if AbsExtPhiSumOfDeviations <= ArcPhaseSyncThrPUInt32:
                SyncingPhaseCylcesCounter += 1
            else:
                SyncingPhaseCylcesCounter = 0
            # Allow a minimum of PHI_STEPS cylces for HoldoffForSync to pass from the moment Rx and Tx frequncies were set
            HoldoffForSync += 1
            if (HoldoffForSync > PHI_STEPS) and (SyncingPhaseCylcesCounter > PARAM_ARC_STAGE2_SYNC_CYCLES):
                SyncedFlag = 1

            # --------------------------------Check Phase Condition-------------------------------- #
            if SyncedFlag == 1:
                if PhaseShiftLargerThanXDegCntr < PARAM_ARC_STAGE2_PHASE_THR_CYLCES:
                    if AbsExtPhiSumOfDeviations > ArcPhaseShiftThrPUInt32:
                        PhaseShiftLargerThanXDegCntr += 1
                    else:
                        PhaseShiftLargerThanXDegCntr = 0
                MaxPhaseShift = max(MaxPhaseShift,AbsExtPhiSumOfDeviations)

            # ------------------------------Check Amplitude Condition------------------------------ #
            # 'Alpha' filter (7/8 & 1/8) the sum of absolute values of the RxSample so that after 32 iterations, the oldest  AbsAmpsSum will be ~ 1%
            AbsAmpsSum = PLL_ABSAMPSSUM_FILTER_COMP_ALPHA * AbsAmpsSum + PLL_ABSAMPSSUM_FILTER_ALPHA * AbsAmpsIncr
            AbsAmpsIncr = 0
            # Get the last 'Amplitude' before power dropping starts
            if before_after_power_down == 0:
                AbsAmpsBefore = AbsAmpsSum
            # Get the last 'Amplitude' before power returned to max output
            if before_after_power_down == 1:
                AbsAmpsAfter = AbsAmpsSum
                if SyncedFlag == 1:
                    ArcPllFinalAmplitudeShift = AbsAmpsAfter - AbsAmpsBefore

        else: # Cross Not Detected
            ExtPhiStep = ExtPhiStepCalc # update step size according to new freq
            ExtPhi += ExtPhiStep

        # We're summarizing abs values instead of actually calculating amplitude due to RT usage consideration.
        # AbsAmpsIncr += abs(sample)
        AbsAmpsIncr += abs(sample)

        # Adding the signals to lists (to plot later...)
        y1.append(SOGI.Y1)
        y2.append(SOGI.Y2)
        x1.append(SOGI.X1)
        abs_amps_sum.append(AbsAmpsSum)
        try:
            sum_of_dev.append(ExtPhiSumOfDeviations*360)
        except:
            sum_of_dev.append(sum_of_dev[-1])
        try:
            current_phi_dev.append(ExtPhiDev*360)
        except:
            current_phi_dev.append(current_phi_dev[-1])

        cycle+=1



    ############# RUN the PLL with the scope_input signal #############

    SOGI = SOGI2(Ki, MainFreq, K1_Constant, ADC_MAX) # Creating the PLL

    i = 0
    for sample in scope_input:
        if i < 0.35 * len(scope_input):
            PLL_FstIterArc_NEW(sample, 0)
        else:
            PLL_FstIterArc_NEW(sample, 1)
        i += 1


    ############# CMD prints #############


    print(f"\nRec file (scope): {file_scope}")
    print("SyncedFlag (Debi method): {}".format(SyncedFlag))
    print("AbsAmpsSum (Debi method): {}\n".format(AbsAmpsSum))

    sum_of_dev = sum_of_dev[plot_cut_after_scope[0]:plot_cut_after_scope[3]]
    abs_amps_sum = abs_amps_sum[plot_cut_after_scope[0]:plot_cut_after_scope[3]]
    phase_shift = max(sum_of_dev) if abs(max(sum_of_dev)) > abs(min(sum_of_dev)) else min(sum_of_dev)
    amplitude_shift = mean(abs_amps_sum[:plot_cut_after_scope[1]]) - mean(abs_amps_sum[plot_cut_after_scope[2]:])
    amplitude_ratio = (1 - (mean(abs_amps_sum[plot_cut_after_scope[2]:])) / mean(abs_amps_sum[:plot_cut_after_scope[1]])) * 100
    print("Method\tPhase shift [°]\tAmplitude Shift\tAmplitude Ratio")
    print(f"Debi calc\t{MaxPhaseShift * 360:.2f}\t{-ArcPllFinalAmplitudeShift:.0f}\t{(1 - AbsAmpsAfter / AbsAmpsBefore) * 100:.2f}")
    print(f"Scope PLL\t{phase_shift:.2f}\t{amplitude_shift:.0f}\t{amplitude_ratio:.2f}")
    if output_plots_dpi[0]:
        if rec_number.lower() in files_spi[file_index].lower():
            fig, axs = plt.subplots(4, sharex=True)
        else:
            fig, axs = plt.subplots(3, sharex=True)
        axs[0].plot(sum_of_dev, "blue", label=f'Scope Phase shift\nMax = {phase_shift:.2f}°')
        axs[1].plot(abs_amps_sum, "blue", label=f'Scope Amplitude\nShift = {amplitude_shift:.0f} [BU]\nRatio = {amplitude_ratio:.2f}%')
        axs[2].plot(scope_input, "c", label='Scope ADC In (RX4)')
        axs[2].plot(y1, "r", label='Scope SOGI.Y1')
    if sys.gettrace() is not None:
        plt.close()
    try:
        dfs = pd.read_csv(path_log_folder + '\\' + files_spi[file_index]).dropna(how='all', axis='columns')
        try:
            if plot_cut_before_spi[0]:
                indexes = dfs[dfs['Machine State'].diff() == plot_cut_before_spi[1]].index.tolist()
                if indexes is None or len(indexes) == 0:
                    print(f'The porper State Machine (={plot_cut_before_spi[1]}) was not found... halving the record')
                    indexes = [int(len(dfs) / 2), int(len(dfs) / 2) + plot_cut_before_spi[3]]
                elif len(indexes) == 1:
                    indexes.append(indexes[0] + plot_cut_before_spi[3])
                else:
                    indexes = indexes[-2:]
                dfs = dfs[indexes[0] + plot_cut_before_spi[2]:indexes[1] + plot_cut_before_spi[2]].reset_index(drop=True)

            sum_of_dev = dfs['Phi Sum of Dev'][plot_cut_after_spi[0]:plot_cut_after_spi[3]].reset_index(drop=True)
            abs_amps_sum = dfs['Abs Amps'][plot_cut_after_spi[0]:plot_cut_after_spi[3]].reset_index(drop=True)
            phase_shift = max(sum_of_dev) if abs(max(sum_of_dev)) > abs(min(sum_of_dev)) else min(sum_of_dev)
            amplitude_shift = mean(abs_amps_sum[:plot_cut_after_scope[1]]) - mean(abs_amps_sum[plot_cut_after_scope[2]:])
            amplitude_ratio = (1 - (mean(abs_amps_sum[plot_cut_after_scope[2]:])) / mean(abs_amps_sum[:plot_cut_after_scope[1]])) * 100
            if output_plots_dpi[0]:
                axs[0].plot(sum_of_dev, "orange", label=f'SPI Phase shift\nMax = {phase_shift:.2f}°')
                axs[1].plot(abs_amps_sum, "orange", label=f'SPI Amplitude\nShift = {amplitude_shift:.0f} [BU]\nRatio = {amplitude_ratio:.2f}%')
                axs[3].plot(dfs['SPI Sample'], "c", label='SPI ADC In (RX4)')
                axs[3].plot(dfs['SOGI.Y1'], "r", label='SPI SOGI.Y1')
            print(f"SPI PLL\t{phase_shift:.2f}\t{amplitude_shift:.0f}\t{amplitude_ratio:.2f}")
        except Exception as error:
            print("An exception occurred:", error)
    except:
        print(f'No SPI file found = {files_spi[file_index]}')
    if output_plots_dpi[0]:
        for ax in axs:
            ax.legend()
        if rec_number in files_spi[file_index]:
            fig.set_size_inches(25, 25)
        else:
            fig.set_size_inches(25, 15)
        plt.savefig(f'{path_log_folder}/PLL {rec_number}.jpg', dpi=output_plots_dpi[1])
        # plt.show()
        plt.close()
    print('Finito')
    print('')


pp = r'V:\HW_Infrastructure\Analog_Team\ARC_Data\Results\Jupiter\Jupiter+ Improved (7E0872F4-EC)\Stage 2 Validation (27-08-2023)\Stage 2 Validation 01 - failed (21-08-2023) - Copy'
ll = ''
fn = 'SEX'
tt = 'sex'
_PC(df, labels=ll, path=pp, file_name=fn, title=tt, auto_open=True)
