# -*- coding: utf-8 -*-
"""
Created on Thu feb 17  2021

@author: Noam.D
"""

##
import os
import time
import pandas as pd
import gc
import NDF_V2

mana = []
tstart = time.time()
from my_pyplot import plot as _P, clear as _PP, print_chrome as _PC

## <codecell> Params to modify

read_all_folder = True   # Reads all the folder content(files ends with .csv)

calc_mag = True
calc_phase = False
plot_on = False
plot_kill = False
scope_meas = False
skip_unimportant_plots = True

if scope_meas:
    Scope_Fs = 1e6
    max_plot_res = 5000
    t_start = 5  # in sec
    t_end = 6    # in sec
else:
    Scope_Fs = 40e3
    max_plot_res = 500000000
    t_start = 0  # in sec
    t_end = 100  # in sec

MH_time = 0.1         # Time for hold
Overlap_time = 0.05
f_resolution = 100    # Spectrum resolution RBW
t_resolution = 0.001  # Time resolution
down_sample = 1
t_stop = t_end
fmin = 1            # in Hz
fmax = 5000000       # in Hz

##########################################


if scope_meas:
    lab_folder = r"M:\Users\Eddy A\Orion\04 Orion Lab F3 (E08EFB51)\Automation\Scope vs Spectrum comparison"
    ch_arr = ["CH1"]
else:   # for SPI
    lab_folder = r"M:\Users\Eddy A\Orion\04 Orion Lab F3 (E08EFB51)\Automation\Scope vs Spectrum comparison v4"
    # ch_arr = ["I-CH1(1)", "Q-CH1(1)", "I-CH2(1)", "Q-CH2(1)", "I-CH3(1)", "Q-CH3(1)"]
    ch_arr = ["I-CH1", "Q-CH1"]



path = lab_folder + '\\'

if not os.path.exists(lab_folder):
    os.makedirs(lab_folder)
test_name = 'Orion lab tests'  # result will be here+filename
file_name_arr = []


if read_all_folder:
    file_arr_temp = []
    file_name_arr += [f for f in os.listdir(path) if (f.endswith('.csv') or f.endswith('.txt')) and 'terminal' not in f]
    for filename in file_name_arr:
        print(filename)
        file_arr_temp += [os.path.splitext(filename)[0]]
    file_name_arr = []
    file_name_arr = file_arr_temp

tag_arr = file_name_arr

print('\nStart Run...')

## <codecell> Read Data Frame from SCOPE or SPI Data CSV file
filename_counter = 0
for filename in file_name_arr:
    if skip_unimportant_plots:
        results_path = path
    else:
        results_path = path + '\\' + filename
    print(filename)
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    if scope_meas:
        df = NDF_V2.scope_CSV_to_df(path, filename, ch_arr, ch_arr, False, Scope_Fs, '.csv', ',', 20)
    else:
        df = NDF_V2.scope_CSV_to_df(path, filename, ch_arr, ch_arr, False, Scope_Fs, '.txt', ',')
    filename_counter += 1

    ## <codecell> DataFrame Chunk

    df1 = NDF_V2.df_Chunk(df, t_start, t_end)
    df = df1
    gc.collect()


    ## <codecell> DataFrame Simple Down Sample

    df1 = df1.iloc[::down_sample]
    Fs = round(1 / (df1['Time'].iloc[1] - df1['Time'].iloc[0]))  # Data samples frequency
    print(Fs)
    ## <codecell> Reset Chunk Time
    df1 = NDF_V2.df_time_reset(df1, 'Time')
    tmax = max(df1['Time'])

    ## <codecell> STFT Spectrogram Calculation of DataFrame


    fft_win = int(Fs / f_resolution)
    fft_win_overlap = int(fft_win - (t_resolution * Fs))

    if fft_win > len(df1):
        fft_win = int(len(df1) / 10)
        fft_win_overlap = int(fft_win * 0.99)
    res_name_arr = []
    (t, f, Zraw, res_name_arr) = NDF_V2.df_stft_RAW_calc(df1, Fs, fft_win, fft_win_overlap, ch_arr, record_type='Orion SPI')
    f = f[f < 250000]

    ## <codecell> Z magnitude calculations
    if calc_mag:
        ZdBm = NDF_V2.Z_mag_calc(Zraw, meas_type='Volts PTP to dBm')

    # <codecell> Z phase calculations
    if calc_phase:
        phase_unwrap = False
        Zphase = NDF_V2.Z_phase_calc(Zraw, phase_unwrap)

    ## <codecell> Plot - Spectrogram
    spectrogram_on = True
    if spectrogram_on:
        for i in range(len(Zraw)):
            meas_sig = res_name_arr[i]
            print(meas_sig)
            if calc_mag:
                if calc_phase:
                    temp_file_name = " 02_mag - "+meas_sig + '_amp'
                else:
                    temp_file_name = filename + " " + meas_sig
                NDF_V2.scpectrogram_plot(ZdBm[i], t, f, max_plot_res, fmax, fmin, t_start,t_stop, plot_on, results_path,
                                         temp_file_name, 2)
            if calc_phase:
                NDF_V2.scpectrogram_plot(Zphase[i], t, f, max_plot_res, fmax, fmin, t_start, t_stop, plot_on,
                                         results_path,
                                         " 02_Phase - " + meas_sig + '_phase', 2)
        if plot_on:
            NDF_V2.kill_chrome(plot_kill)

    ## <codecell> Sliding Window Magnitude Spectrum Analysis and Plots
    if calc_mag:

        win_size = int(MH_time / t_resolution)
        win_overlap = int(Overlap_time / t_resolution)
        for channel in ch_arr:
            print(channel)
            df_MH_1 = pd.DataFrame(columns=['f'])
            df_AVG_1 = pd.DataFrame(columns=['f'])

            df_MH_1['f'] = f
            df_AVG_1['f'] = f

            indices = [i for i, elem in enumerate(res_name_arr) if channel in elem]
            print(indices)
            if not indices:
                continue
            for i in indices:
                meas_sig = res_name_arr[i]
                if channel in meas_sig:
                    (df_MH_temp, df_AVG_temp, t_list) = NDF_V2.sliding_spectrum(ZdBm[i], t, f, win_size, win_overlap,
                                                                             meas_sig)

                df_MH_1[df_MH_temp.columns] = df_MH_temp
                df_AVG_1[df_AVG_temp.columns] = df_AVG_temp

            del df_MH_temp
            del df_AVG_temp
            scope_plot_data = NDF_V2.data_plot(df1, 'Scope RAW Data', 'Time', max_plot_res, channel,
                                               plot_on, not skip_unimportant_plots, results_path)
            MH1_plot_data = NDF_V2.data_plot(df_MH_1, 'Sliding Spectrum MH', 'f', max_plot_res, channel,
                                             plot_on, not skip_unimportant_plots, results_path)
            AVG1_plot_data = NDF_V2.data_plot(df_AVG_1, 'Sliding Spectrum AVG', 'f', max_plot_res, channel,
                                              plot_on, not skip_unimportant_plots, results_path)

            data_list = [scope_plot_data, MH1_plot_data]
            name_list = ['ZeroSpan Results', 'Sliding MH Spectrum dBm']
            # data_list=[ MH1_plot_data]
            # name_list=['Sliding MH Spectrum dBm']
            x_sync = False
            if not skip_unimportant_plots:
                data_name = '03_mag - Sliding FFT MH Spectrum with RAW'
                NDF_V2.data_pane_plot(data_name, data_list, name_list, plot_on, x_sync, tag_arr, channel, results_path, 2)

                data_list = [scope_plot_data, AVG1_plot_data]
                name_list = ['ZeroSpan Results', 'Sliding AVG Spectrum dBm']
                # data_list=[ AVG1_plot_data]
                # name_list=['Sliding AVG Spectrum dBm']
                x_sync = False
                data_name = '04_mag - Sliding FFT AVG Spectrum with RAW '
                NDF_V2.data_pane_plot(data_name, data_list, name_list, plot_on, x_sync, tag_arr, channel, results_path, 2)

            data_list = [MH1_plot_data, AVG1_plot_data]
            name_list = ['Sliding MH Spectrum dBm', 'Sliding AVG Spectrum dBm']
            x_sync = False
            if calc_phase:
                data_name = '05_mag - Sliding FFT MH and AVG Spectrum'
            else:
                data_name = filename + " " + 'Sliding FFT'

            NDF_V2.data_pane_plot(data_name, data_list, name_list, plot_on, x_sync, tag_arr, channel, results_path, 2)
        NDF_V2.kill_chrome(plot_kill)
    ## <codecell> Sliding Window Phase Spectrum Analysis and Plots
    slide_FFT_phase = True;
    if calc_phase and slide_FFT_phase:
        win_size = int(MH_time / t_resolution)
        win_overlap = int(Overlap_time / t_resolution)

        for channel in ch_arr:
            df_MH_1 = pd.DataFrame(columns=['f'])
            df_AVG_1 = pd.DataFrame(columns=['f'])

            df_MH_1['f'] = f
            df_AVG_1['f'] = f

            indices = [i for i, elem in enumerate(res_name_arr) if channel in elem]

            for i in indices:
                meas_sig = res_name_arr[i]
                if channel in meas_sig:
                    (df_MH_temp, df_AVG_temp, t_list) = NDF_V2.sliding_spectrum(Zphase[i], t, f, win_size, win_overlap,
                                                                             meas_sig)

                df_MH_1[df_MH_temp.columns] = df_MH_temp
                df_AVG_1[df_AVG_temp.columns] = df_AVG_temp

            file_on = True
            scope_plot_data = NDF_V2.data_plot(df1, 'Scope RAW Data', 'Time', max_plot_res, channel,
                                               plot_on, not skip_unimportant_plots, results_path)
            MH1_plot_data = NDF_V2.data_plot(df_MH_1, 'Sliding Spectrum MH', 'f', max_plot_res, channel,
                                             plot_on, not skip_unimportant_plots, results_path)
            AVG1_plot_data = NDF_V2.data_plot(df_AVG_1, 'Sliding Spectrum AVG', 'f', max_plot_res, channel,
                                              plot_on, not skip_unimportant_plots, results_path)

            data_list = [scope_plot_data, MH1_plot_data]
            name_list = ['Scope RAW', 'Sliding MH Spectrum dBm']
            x_sync = False
            data_name = '03_phase - Sliding FFT MH Spectrum with Scope RAW'
            NDF_V2.data_pane_plot(data_name, data_list, name_list, plot_on, x_sync, tag_arr, channel, results_path, 2)

            data_list = [scope_plot_data, AVG1_plot_data]
            name_list = ['Scope RAW', 'Sliding AVG Spectrum dBm']
            x_sync = False
            data_name = '04_phase - Sliding FFT AVG Spectrum with Scope RAW'
            NDF_V2.data_pane_plot(data_name, data_list, name_list, plot_on, x_sync, tag_arr, channel, results_path, 2)

            data_list = [MH1_plot_data, AVG1_plot_data]
            name_list = ['Sliding MH Spectrum dBm', 'Sliding AVG Spectrum dBm']
            x_sync = True
            data_name = '05_phase - Sliding FFT MH and AVG Spectrum'
            NDF_V2.data_pane_plot(data_name, data_list, name_list, plot_on, x_sync, tag_arr, channel, results_path, 2)

    NDF_V2.kill_chrome(plot_kill)

    # <codecell> End Sequence

