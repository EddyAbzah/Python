import os
import math
import plotly
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from scipy import signal
from statistics import mean
from plotly.subplots import make_subplots
from my_pyplot import omit_plot as _O, plot as _P, print_chrome as _PC, clear as _PP, print_lines as _PL, send_mail as _SM
import Plot_Graphs_with_Sliders as _G
import my_tools


# ####   Files   #### #
scope_columns = ['PLL', 'SPI_Sample']
spi_columns = {0: 'SPI_Sample', 1: 'SOGI.Y1', 2: 'Phi_Current_Dev', 3: 'Phi_Sum_of_Dev', 4: 'Abs_Amps', 5: 'Machine_State', 6: 'arcer'}
path_log_folder = r'V:\HW_Infrastructure\Analog_Team\ARC_Data\Results\Jupiter\Jupiter+ Improved 2024 (7B0F73A7-A4)\JPI Stage 2 Validation' + '\\'
path_log_folder += 'Stage 2 Validation 01 (15-01-2024)'
# path_log_folder += 'Stage 2 Validation 02 (15-01-2024)'
path_output_folder = path_log_folder + r'\PLLs'
filter_rec = ['']
path_log_name_scope = ['scope rec', '.csv']
path_log_name_spi = ['spi rec', '.csv']
path_log_name_mngr = ['sedsp mngr', '.log', '[15] Arc stage2', 'Event Params:']
path_log_name_pwr = ['sedsp pwr', '.log', 'Ev15 Struct:']
files_scope = sorted([f for f in os.listdir(path_log_folder) if all(s in f.lower() for s in path_log_name_scope) and any(s in f.lower() for s in filter_rec)])
files_spi = sorted([f for f in os.listdir(path_log_folder) if all(s in f.lower() for s in path_log_name_spi) and any(s in f.lower() for s in filter_rec)])
files_inverter_mngr = sorted([f for f in os.listdir(path_log_folder) if all(s in f.lower() for s in path_log_name_mngr[:2]) and any(s in f.lower() for s in filter_rec)])
files_inverter_pwr = sorted([f for f in os.listdir(path_log_folder) if all(s in f.lower() for s in path_log_name_pwr[:2]) and any(s in f.lower() for s in filter_rec)])

# ####   True   ###   False   #### #
output_html_auto_open = [True, True]
terminal_print_summary = True
convert_scope_input_to_adc = [False, lambda x: x * 2**14 / 2.5 - 8200]
adjust_scope_input_level_scope = [False, 1.03]
decimate_scope_input = [False, 3]   # from Fs = 50kHz to 16.667kHz or 12.5kHz
plot_cut_before_scope = [False, 5000, 25000]
plot_level_before_scope = [True, 350]
# plot_cut_after_scope = [True, 2500, -1]
plot_cut_after_scope = [True, 2500, int(17e3)]
plot_cut_for_amplitude = [1700, 12000]
plot_cut_before_spi = [True, 1, -2100, 92200, 21000]        # 0 = ToF, 1 = State Machine, 2 = start trim, 3 = desired length
# plot_cut_after_spi = [True, 2500, -1]
plot_cut_after_spi = [True, 2500, int(17e3)]
expected_length_seconds = [False, 0, 20000]

# ####   PLL   #### #
SysCLK = 16667
MainFreq = 500
Ki = 0.03  # Old Ki = 0.045
KpCoeff = 1
correction_factor = 0.99996   # OLD = 1.00012

############### Yalla ###############
if terminal_print_summary:
    print("Record\tMethod\tPhase shift [°]\tAmplitude Shift\tAmplitude Ratio")
for file_index in range(max(len(files_scope), len(files_spi))):
    try:
        file_scope = files_scope[file_index]
        rec_number = file_scope[file_scope.lower().find('rec'):file_scope.lower().find('rec') + 6]
        if 'ka2' in file_scope.lower():
            MainFreq = 492.738281
        else:
            MainFreq = 511.402344
    except:
        file_spi = files_spi[file_index]
        rec_number = file_spi[file_spi.lower().find('rec'):file_spi.lower().find('rec') + 6]
        file_scope = None
        if 'ka2' in file_spi.lower():
            MainFreq = 492.738281
        else:
            MainFreq = 511.402344
    y_axis_titles = ['Method', 'Phase [°]', 'Amp shift', 'Amp ratio']

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
        plot_rows = int(len(df.columns) / 1)
        scope_input = None
        for index_df, (title_df, sub_df) in enumerate(df.items()):
            if title_df == scope_columns[0] or title_df == scope_columns[1]:  # SPI_Sample    SOGI.Y1
                scope_input = sub_df
        if convert_scope_input_to_adc[0]:
            scope_input = convert_scope_input_to_adc[1](scope_input)
        if decimate_scope_input[0]:
            scope_input = signal.decimate(scope_input, decimate_scope_input[1])
        if adjust_scope_input_level_scope[0]:
            scope_input = (scope_input - scope_input.mean()) * adjust_scope_input_level_scope[1]
        if plot_cut_before_scope[0]:
            scope_input = scope_input[plot_cut_before_scope[1]:plot_cut_before_scope[2]]
        if plot_level_before_scope[0]:
            scope_input = [i for i in scope_input if abs(i) > plot_level_before_scope[1]]
        if expected_length_seconds[0]:
            scope_input = scope_input[expected_length_seconds[1]:expected_length_seconds[2]]
    except:
        if not terminal_print_summary:
            print(f'ERROR!!! file "{path_log_folder}/{file_scope}" not found')


    def parse_text_mngr(line):
        try:
            temp = [float(f) for f in line[line.find('s:') + 2:line.find('\n')].split(' ') if f.replace("-", "").replace(".", "").isnumeric()]
            if len(temp) < 3:
                return [False]
            if len(temp) < 5:
                temp = [*temp, *[0] * (5 - len(temp))]
            return [True, *temp]
        except:
            return [False]

    def parse_text_pwr(line):
        try:
            temp = [float(f) for f in line.replace(",", "").split(' ') if f.replace("-", "").replace(".", "").isnumeric()]
            if len(temp) < 3:
                return [False]
            if len(temp) < 5:
                temp = [*temp, *[0] * (5 - len(temp))]
            return [True, *temp]
        except:
            return [False]


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
        cycle += 1


    ############# RUN the PLL with the scope_input signal #############
    SOGI = SOGI2(Ki, MainFreq, K1_Constant, ADC_MAX) # Creating the PLL
    if file_scope is not None:
        i = 0
        for sample in scope_input:
            if i < 0.325 * len(scope_input):
                PLL_FstIterArc_NEW(sample, 0)
            else:
                PLL_FstIterArc_NEW(sample, 1)
            i += 1


############# Scope prints #############
        if not terminal_print_summary:
            print(f"\nRec file (scope): {file_scope}")
            print("SyncedFlag (Debi method): {}".format(SyncedFlag))
            print("AbsAmpsSum (Debi method): {}\n".format(AbsAmpsSum))
        if plot_cut_after_scope[0]:
            sum_of_dev = sum_of_dev[plot_cut_after_scope[1]:plot_cut_after_scope[2]]
            abs_amps_sum = abs_amps_sum[plot_cut_after_scope[1]:plot_cut_after_scope[2]]
            scope_input = scope_input[plot_cut_after_scope[1]:plot_cut_after_scope[2]]
            y1 = y1[plot_cut_after_scope[1]:plot_cut_after_scope[2]]
        phase_shift = max(sum_of_dev) if abs(max(sum_of_dev)) > abs(min(sum_of_dev)) else min(sum_of_dev)
        amplitude_shift = mean(abs_amps_sum[:plot_cut_for_amplitude[0]]) - mean(abs_amps_sum[plot_cut_for_amplitude[1]:])
        amplitude_ratio = (1 - (mean(abs_amps_sum[plot_cut_for_amplitude[1]:])) / mean(abs_amps_sum[:plot_cut_for_amplitude[0]])) * 100
        if not terminal_print_summary:
            print("Method\tPhase shift [°]\tAmplitude Shift\tAmplitude Ratio")
        print(f"{rec_number}\tDebi calc\t{MaxPhaseShift * 360:.2f}\t{-ArcPllFinalAmplitudeShift:.0f}\t{(1 - AbsAmpsAfter / AbsAmpsBefore) * 100:.2f}")
        print(f"{rec_number}\tScope Python\t{phase_shift:.2f}\t{amplitude_shift:.0f}\t{amplitude_ratio:.2f}")
        if output_html_auto_open[0]:
            fig = make_subplots(rows=4, cols=1, shared_xaxes=True, subplot_titles=['Phase shift [°]', 'Amplitude [BU]', 'RX out [ADC]', 'SOGI.Y1 [ADC]'])
            y_axis_titles = [a + ' ' * (10 + len(max(y_axis_titles)) - len(a)) + b for a, b in zip(y_axis_titles, ['Scope Py', f'{phase_shift:.2f}', f'{amplitude_shift:.0f}', f'{amplitude_ratio:.2f}'])]
            y_axis_titles = [a + ' ' * (10 + len(max(y_axis_titles)) - len(a)) + b for a, b in zip(y_axis_titles, ['Scope Debi', f'{MaxPhaseShift * 360:.2f}', f'{-ArcPllFinalAmplitudeShift:.0f:.0f}', f'{(1 - AbsAmpsAfter / AbsAmpsBefore) * 100:.2f}'])]
            fig.add_trace(go.Scatter(y=sum_of_dev, name=f'Scope Python - Phase shift (max = {phase_shift:.2f}°)'), col=1, row=1)
            fig.add_trace(go.Scatter(y=abs_amps_sum, name=f'Scope Python - Amplitude (shift = {amplitude_shift:.0f}[BU]   Ratio = {amplitude_ratio:.2f}%)'), col=1, row=2)
            fig.add_trace(go.Scatter(y=scope_input, name='Scope Python - ADC In'), col=1, row=3)
            fig.add_trace(go.Scatter(y=y1, name='Scope Python - SOGI.Y1'), col=1, row=4)
    else:
        fig = make_subplots(rows=4, cols=1, shared_xaxes=True, subplot_titles=['Phase shift [°]', 'Amplitude [BU]', 'RX out [ADC]', 'SOGI.Y1 [ADC]'])


############# SPI prints #############
    try:
        fig_title = 'Python PLL ' + files_spi[file_index][:-4]
        dfs = pd.read_csv(path_log_folder + '\\' + files_spi[file_index]).dropna(how='all', axis='columns')
        if plot_cut_before_spi[0]:
            indexes = dfs[dfs[spi_columns[5]].diff() == plot_cut_before_spi[1]].index.tolist()
            if indexes is None or len(indexes) == 0:
                print(f'The porper State Machine (={plot_cut_before_spi[1]}) was not found... halving the record')
                indexes = [int(len(dfs) / 2.5), int(len(dfs) / 2.5) + plot_cut_before_spi[3]]
            elif len(indexes) == 1:
                indexes.append(indexes[0] + plot_cut_before_spi[3])
            else:
                indexes = indexes[-2:]
            dfs = dfs[indexes[0] + plot_cut_before_spi[2]:indexes[1] + plot_cut_before_spi[2]].reset_index(drop=True)
            dfs = dfs[:plot_cut_before_spi[4]].reset_index(drop=True)
        if plot_cut_after_spi[0]:
            scope_input = dfs[spi_columns[0]][plot_cut_after_scope[1]:plot_cut_after_scope[2]].reset_index(drop=True)
            y1 = dfs[spi_columns[1]][plot_cut_after_scope[1]:plot_cut_after_scope[2]].reset_index(drop=True)
            sum_of_dev = dfs[spi_columns[3]][plot_cut_after_spi[1]:plot_cut_after_spi[2]].reset_index(drop=True)
            abs_amps_sum = dfs[spi_columns[4]][plot_cut_after_spi[1]:plot_cut_after_spi[2]].reset_index(drop=True)
        else:
            scope_input = dfs[spi_columns[0]]
            y1 = dfs[spi_columns[1]]
            sum_of_dev = dfs[spi_columns[3]]
            abs_amps_sum = dfs[spi_columns[4]]

        phase_shift = max(sum_of_dev) if abs(max(sum_of_dev)) > abs(min(sum_of_dev)) else min(sum_of_dev)
        amplitude_shift = mean(abs_amps_sum[:plot_cut_for_amplitude[0]]) - mean(abs_amps_sum[plot_cut_for_amplitude[1]:])
        amplitude_ratio = (1 - (mean(abs_amps_sum[plot_cut_for_amplitude[1]:])) / mean(abs_amps_sum[:plot_cut_for_amplitude[0]])) * 100
        if output_html_auto_open[0]:
            y_axis_titles = [a + ' ' * (10 + len(max(y_axis_titles)) - len(a)) + b for a, b in zip(y_axis_titles, ['SPI RAW', f'{phase_shift:.2f}', f'{amplitude_shift:.0f}', f'{amplitude_ratio:.2f}'])]
            fig.add_trace(go.Scatter(y=sum_of_dev, name=f'SPI RAW - Phase shift (max = {phase_shift:.2f}°)'), col=1, row=1)
            fig.add_trace(go.Scatter(y=abs_amps_sum, name=f'SPI RAW - Amplitude (shift = {amplitude_shift:.0f}[BU]   Ratio = {amplitude_ratio:.2f}%)'), col=1, row=2)
            fig.add_trace(go.Scatter(y=scope_input, name='SPI RAW - ADC In'), col=1, row=3)
            fig.add_trace(go.Scatter(y=y1, name='SPI RAW - SOGI.Y1'), col=1, row=4)
        if not terminal_print_summary:
            print("Record\tMethod\tPhase shift [°]\tAmplitude Shift\tAmplitude Ratio")
        print(f"{rec_number}\tSPI RAW\t{phase_shift:.2f}\t{amplitude_shift:.0f}\t{amplitude_ratio:.2f}")

        y1 = []
        y2 = []
        x1 = []
        sum_of_dev = [0]
        abs_amps_sum = []
        current_phi_dev = [0]
        SOGI = SOGI2(Ki, MainFreq, K1_Constant, ADC_MAX)  # Creating the PLL
        i = 0
        for sample in dfs[spi_columns[0]]:
            if i < 0.325 * len(dfs[spi_columns[0]]):
                PLL_FstIterArc_NEW(sample, 0)
            else:
                PLL_FstIterArc_NEW(sample, 1)
            i += 1
        if plot_cut_after_scope[0]:
            sum_of_dev = sum_of_dev[plot_cut_after_scope[1]:plot_cut_after_scope[2]]
            abs_amps_sum = abs_amps_sum[plot_cut_after_scope[1]:plot_cut_after_scope[2]]
            y1 = y1[plot_cut_after_scope[1]:plot_cut_after_scope[2]]
        phase_shift = max(sum_of_dev) if abs(max(sum_of_dev)) > abs(min(sum_of_dev)) else min(sum_of_dev)
        amplitude_shift = mean(abs_amps_sum[:plot_cut_for_amplitude[0]]) - mean(abs_amps_sum[plot_cut_for_amplitude[1]:])
        amplitude_ratio = (1 - (mean(abs_amps_sum[plot_cut_for_amplitude[1]:])) / mean(abs_amps_sum[:plot_cut_for_amplitude[0]])) * 100
        print(f"{rec_number}\tDebi calc\t{MaxPhaseShift * 360:.2f}\t{-ArcPllFinalAmplitudeShift:.0f}\t{(1 - AbsAmpsAfter / AbsAmpsBefore) * 100:.2f}")
        print(f"{rec_number}\tSPI Python\t{phase_shift:.2f}\t{amplitude_shift:.0f}\t{amplitude_ratio:.2f}")
        if output_html_auto_open[0]:
            y_axis_titles = [a + ' ' * (10 + len(max(y_axis_titles)) - len(a)) + b for a, b in zip(y_axis_titles, ['SPI Py', f'{phase_shift:.2f}', f'{amplitude_shift:.0f}', f'{amplitude_ratio:.2f}'])]
            y_axis_titles = [a + ' ' * (10 + len(max(y_axis_titles)) - len(a)) + b for a, b in zip(y_axis_titles, ['SPI Debi', f'{MaxPhaseShift * 360:.2f}', f'{-ArcPllFinalAmplitudeShift:.0f}', f'{(1 - AbsAmpsAfter / AbsAmpsBefore) * 100:.2f}'])]
            fig.add_trace(go.Scatter(y=sum_of_dev, name=f'SPI Python - Phase shift (max = {phase_shift:.2f}°)'), col=1, row=1)
            fig.add_trace(go.Scatter(y=abs_amps_sum, name=f'SPI Python - Amplitude (shift = {amplitude_shift:.0f}[BU]   Ratio = {amplitude_ratio:.2f}%)'), col=1, row=2)
            fig.add_trace(go.Scatter(y=y1, name='SPI Python - SOGI.Y1'), col=1, row=4)
    except:
        print(f'OR!!!\tNo SPI file found = {files_spi[file_index]}')

### Get Event from MNGR:
    try:
        fig_title = 'Python PLL ' + files_inverter_mngr[file_index][files_inverter_mngr[file_index].lower().find('mngr') + 5:-4]
        if not terminal_print_summary:
            temp = files_inverter_mngr[file_index][files_inverter_mngr[file_index].lower().find('(') + 1:files_inverter_mngr[file_index].lower().find(')')]
            print(f"{rec_number}\tPLC\t" + "{}\tScenario\t{}".format(*temp.split(' ')))
        parse_text_output = [False]
        with open(path_log_folder + '\\' + files_inverter_mngr[file_index], 'r') as file:
            lines = file.readlines()
            for line in lines:
                if line.find(path_log_name_mngr[2]) != -1:
                    if parse_text_output[0]:
                        print(f'ERROR!!!\t"{path_log_name_mngr[2]}" is already found in this file.')
                    else:
                        for l in lines[lines.index(line):lines.index(line) + 6]:
                            if l.find(path_log_name_mngr[3]) != -1:
                                parse_text_output = parse_text_mngr(l)
        if parse_text_output[0]:
            print(f"{rec_number}\tMNGR prints\t{parse_text_output[3]:.2f}\t{parse_text_output[4]:.0f}\t{parse_text_output[5]:.2f}")
            y_axis_titles = [a + ' ' * (10 + len(max(y_axis_titles)) - len(a)) + b for a, b in zip(y_axis_titles, ['MNGR', f'{parse_text_output[3]:.2f}', f'{parse_text_output[4]:.0f}', f'{parse_text_output[5]:.2f}'])]
        else:
            print(f"{rec_number}\tMNGR prints\tN/A\tN/A\tN/A")
    except:
        print(f'ERROR!!!\tNo MNGR file found for file_index = {file_index}')

### Get Event from PWR:
    try:
        fig_title = 'Python PLL ' + files_inverter_pwr[file_index][files_inverter_pwr[file_index].lower().find('pwr') + 4:-4]
        if not terminal_print_summary:
            temp = files_inverter_pwr[file_index][files_inverter_pwr[file_index].lower().find('(') + 1:files_inverter_pwr[file_index].lower().find(')')]
            print(f"{rec_number}\tPLC\t" + "{}\tScenario\t{}".format(*temp.split(' ')))
        parse_text_output = [False]
        with open(path_log_folder + '\\' + files_inverter_pwr[file_index], 'r') as file:
            lines = file.readlines()
            for line in lines:
                if line.find(path_log_name_pwr[2]) != -1:
                    if parse_text_output[0]:
                        print(f'ERROR!!!\t"{path_log_name_pwr[2]}" is already found in this file.')
                    else:
                        parse_text_output = parse_text_pwr(line)
        if parse_text_output[0]:
            print(f"{rec_number}\tPWR prints\t{parse_text_output[3]:.2f}\t{parse_text_output[4]:.0f}\t{parse_text_output[5]:.2f}")
            y_axis_titles = [a + ' ' * (10 + len(max(y_axis_titles)) - len(a)) + b for a, b in zip(y_axis_titles, ['PWR', f'{parse_text_output[3]:.2f}', f'{parse_text_output[4]:.0f}', f'{parse_text_output[5]:.2f}'])]
        else:
            print(f"{rec_number}\tPWR prints\tN/A\tN/A\tN/A")
    except:
        print(f'ERROR!!!\tNo PWR file found file_index = {file_index}')

    if output_html_auto_open[0]:
        fig.update_layout(title=fig_title, title_font_color="#407294", title_font_size=40, legend_title="Plots:")
        plotly.offline.plot(fig, config={'scrollZoom': True, 'editable': True}, filename=f'{path_output_folder}/{fig_title}.html', auto_open=output_html_auto_open[1])
    if not terminal_print_summary:
        print('Finito\n')
