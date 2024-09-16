# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 11:26:26 2020

@author: Noam.D
"""

## <codecell> Improts

import os
import time
import pandas as pd
import igf
import igf2
import gc
import ndf
import sys
import pywt
import numpy as np
import plotly.express as px
import plotly as py
import plotly.graph_objs as go
import math
import statistics
import scipy.fftpack
from scipy import signal
import matplotlib.pyplot as plt
import decida
import webbrowser

F = 2e6

W = '\033[0m'  # white (normal)
R = '\033[31m'  # red
G = '\033[32m'  # green
O = '\033[33m'  # orange
B = '\033[34m'  # blue
P = '\033[35m'  # purple

tstart = time.time()

# ==================optimizer modes====================================
# buck_DCM = 0
# boost_DCM = 1
# BuckBoost_DCM = 2
# buck_CCM = 9
# boost_CCM = 10
# BuckBoost_CCM =11

# The frequency in buck and boost modes is 200KHz
# The frequency in BuckBoost mode is 133KHz

# =============================================================================


print('debug1')

## <codecell> Directory, Files, Tags, Data location

read_all_folder = True
lab_folder = r'C:\Users\eddy.a\Documents\Python Scripts\Arc Analysis and Model\Log Files'

# lab_folder='LV450 ARC Measurements - OLD'
path = lab_folder + '\\'

if not os.path.exists(lab_folder):
    os.makedirs(lab_folder)
test_name = 'ARC analysis'
file_name_arr = []
file_name_a1 = []
# file_name_arr = ['arc 1 SPI','arc 2 SPI','arc 3 SPI','clean no telem 1 SPI','clean no telem 2 SPI','rst1 SPI','rst2 SPI','rst3 SPI','telem1 SPI','telem2 SPI','telem3 SPI','slow arc 0.25 2 SPI','slow arc 0.25 SPI']
# tag_arr =  ['arc 1 SPI','arc 2 SPI','arc 3 SPI','clean no telem 1 SPI','clean no telem 2 SPI','rst1 SPI','rst2 SPI','rst3 SPI','telem1 SPI','telem2 SPI','telem3 SPI','slow arc 0.25 2 SPI','slow arc 0.25 SPI']


# tag_arr =  ['Time',	'LRX',	'RXOUT',	'I_IN_AC',	'I_IN_DC']

tag_arr = ['LRX']

if read_all_folder:
    file_arr_temp = []
    file_name_arr += [f for f in os.listdir(path) if f.endswith('.csv')]
    for filename in file_name_arr:
        print(filename)
        file_arr_temp += [os.path.splitext(filename)[0]]
    file_name_arr = []
    file_name_arr = file_arr_temp

# file_name_arr = ['arc 2 SPI']
tag_arr = file_name_arr
import os

spi_ch_arr = ['VRX']
scope_ch_arr = ['Time', 'LRX', 'RXOUT', 'I_IN_AC', 'I_IN_DC']
# spi_ch_arr = ['Vin']
# spi_ch_arr = ['Vin','Vout','Iin','IL']
results_path = path + test_name
# spi_ch_arr = ['Rx']
if not os.path.exists(results_path):
    os.makedirs(results_path)

print('\nStart Run...')

## <codecell> Read Data Frame from SCOPE or SPI Data CSV file
filename_counter = 0
for filename in file_name_arr:
    results_path = path + test_name + '\\' + filename
    print(B + filename + W)
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    scope_file_name_arr = [s for s in file_name_arr if "SCOPE" in s]
    scope_tag_arr = [s for s in tag_arr if "SCOPE" in s]

    spi_file_name_arr = [s for s in file_name_arr if "SPI" in s]
    spi_tag_arr = [s for s in tag_arr if "SPI" in s]
    if len(scope_file_name_arr) > 0:
        ch_arr = scope_ch_arr
        df = igf.scope_CSV_to_df(path, scope_file_name_arr, scope_ch_arr, scope_ch_arr)
        print(df.head)
    if len(spi_file_name_arr) > 0:
        spi_Fs = F
        spi_filename = filename
        ch_arr = spi_ch_arr
        df = igf.spi_TXT_to_df(path, spi_filename, spi_tag_arr[filename_counter], spi_ch_arr, spi_Fs)
        print(df.head)
    filename_counter += 1

    #    df.to_csv(results_path +' df.csv')
    # df2=df
    # df2=df2.drop('Time', axis=1, inplace=False)
    # df2['Ratio']=df2.iloc[:,1:2].values/df2.iloc[:,0:1].values ##Vout/Vin
    ## <codecell> DataFrame Chunk
    t_start = 0
    t_end = 100
    df1 = igf.df_Chunk(df, t_start, t_end)
    gc.collect()

    var_win_factor = 100
    avg_win_factor = 20

    ## <codecell> DataFrame Simple Down Sample
    down_sample = 1
    df1 = df1.iloc[::down_sample]
    Fs = round(1 / (df1['Time'].iloc[1] - df1['Time'].iloc[0]))  # Data samples frequency
    print(Fs)

    ## <codecell> STFT Spectrogram Calculation of DataFrame

    f_resolution = 100  # Spectrum resolution RBW
    t_resolution = 0.001  # Time resolution

    fft_win = int(Fs / f_resolution)
    fft_win_overlap = int(fft_win - (t_resolution * Fs))
    # fft_win = 10000
    # fft_win_overlap = 5000

    if fft_win > len(df1):
        fft_win = int(len(df1) / 10)
        fft_win_overlap = int(fft_win * 0.99)
    res_name_arr = []
    (t, f, Zraw, res_name_arr) = igf.df_stft_RAW_calc(df1, Fs, fft_win, fft_win_overlap, ch_arr)
    f = f[f < 200000]

    ## <codecell> Z magnitude calculations

    calc_mag = True
    calc_phase = True
    if calc_mag:
        ZdBm = igf.Z_mag_calc(Zraw)

    # <codecell> Z phase calculations
    if calc_phase:
        phase_unwrap = False
        Zphase = igf.Z_phase_calc(Zraw, phase_unwrap)

    ## <codecell> Plot - Spectrogram
    spectrogram_on = False
    if spectrogram_on:
        fmin = 0
        fmax = 150000
        t_start = 0
        max_plot_res = 5000
        plot_on = True
        for i in range(len(Zraw)):
            meas_sig = res_name_arr[i]
            print(O + meas_sig + W)
            if calc_mag:
                igf.scpectrogram_plot(ZdBm[i], t, f, max_plot_res, fmax, fmin, t_start, plot_on, results_path,
                                      meas_sig + '_amp')
            if calc_phase:
                igf.scpectrogram_plot(Zphase[i], t, f, max_plot_res, fmax, fmin, t_start, plot_on, results_path,
                                      meas_sig + '_phase')
                os.system("taskkill /im chrome.exe /f")

    ## <codecell> ZeroSpan results for STFT results
    save_zero_span = True
    calc_phase = True
    if calc_mag:
        # zero_span_arr=[53e3, 106e3, 159e3]
        zero_span_arr = [106e3]
        df_fft = igf.ZeroSpan_calc(ZdBm, res_name_arr, t, f, zero_span_arr, ch_arr)
        zs_amp_str = str(zero_span_arr).replace(",", " -")
        ndf.Save_df_fft_mag(results_path, df_fft, filename)

    if calc_phase:
        zero_span_phase_arr = [106e3]
        df_fft_phase = igf.ZeroSpan_calc(Zphase, res_name_arr, t, f, zero_span_phase_arr, ch_arr)
        cols = df_fft_phase.columns[1:]
        df_fft_phase[cols] = df_fft_phase[cols].diff()
        zs_phase_str = str(zero_span_phase_arr).replace(",", " -")
        ndf.Save_df_fft_phase(results_path, df_fft_phase, filename)

    ##
    max_plot_res = 100000
    calc_mag = True
    if calc_mag:
        # for channel in ch_arr:
        avg_win_t = 0.1
        var_win_t = 0.05
        win_size = int(var_win_t / t_resolution)
        fft_var = igf.df_var_calc(df_fft, win_size, 't')

        win_size = int(avg_win_t / t_resolution)
        fft_var_mean = igf.df_mean_calc(fft_var, win_size, 't')
        new_col = fft_var_mean.columns[1:]
        fft_var[new_col] = fft_var_mean[new_col]

        fft_mean = igf.df_mean_calc(df_fft, win_size, 't')
        new_col = fft_mean.columns[1:]
        df_fft[new_col] = fft_mean[new_col]

        del fft_var_mean
        del fft_mean
        gc.collect()

        for channel in ch_arr:
            file_on = False
            plot_on = False
            scope_plot_data = igf.data_plot(df1, 'SPI RAW Data', 'Time', max_plot_res, channel, plot_on, file_on,
                                            results_path)
            fft_plot_data = igf.data_plot(df_fft, 'Zero Span FFT', 't', max_plot_res, channel, plot_on, file_on,
                                          results_path)
            fft_var_plot_data = igf.data_plot(fft_var, 'Zero Span FFT Variance', 't', max_plot_res, channel, plot_on,
                                              file_on, results_path)

            plot_on = True
            data_list = [scope_plot_data, fft_plot_data]
            name_list = ['RAW Data', 'ZeroSpan FFT']
            data_list = [scope_plot_data, fft_plot_data, fft_var_plot_data]
            name_list = ['RAW Data', 'ZeroSpan FFT', 'ZeroSpan FFT Variance']
            x_sync = True
            igf.data_pane_plot('02_mag - FFT Magnitude Zero Span Plots for ' + zs_amp_str + ' Hz freqs', data_list,
                               name_list, plot_on, x_sync, tag_arr, channel, results_path)

    max_plot_res = 100000
    if calc_phase:
        # for channel in ch_arr:
        var_win_t = 0.05
        win_size = int(var_win_t / t_resolution)
        fft_var = igf.df_var_calc(df_fft_phase, win_size, 't')

        avg_win_t = 0.1
        win_size = int(avg_win_t / t_resolution)
        fft_var_mean = igf.df_mean_calc(fft_var, win_size, 't')
        new_col = fft_var_mean.columns[1:]
        fft_var[new_col] = fft_var_mean[new_col]

        fft_mean = igf.df_mean_calc(df_fft_phase, win_size, 't')
        new_col = fft_mean.columns[1:]
        df_fft_phase[new_col] = fft_mean[new_col]

        del fft_var_mean
        del fft_mean
        gc.collect()
        for channel in ch_arr:
            file_on = False
            plot_on = False
            scope_plot_data = igf.data_plot(df1, 'Scope RAW Data', 'Time', max_plot_res, channel, plot_on, file_on,
                                            results_path)
            fft_phase_plot_data = igf.data_plot(df_fft_phase, 'Zero Span FFT', 't', max_plot_res, channel, plot_on,
                                                file_on, results_path)
            fft_var_plot_data = igf.data_plot(fft_var, 'Zero Span FFT Variance', 't', max_plot_res, channel, plot_on,
                                              file_on, results_path)

            plot_on = True
            data_list = [scope_plot_data, fft_phase_plot_data, fft_var_plot_data]
            name_list = ['RAW Data', 'ZeroSpan Phase', 'ZeroSpan Phase Variance']
            x_sync = True
            igf.data_pane_plot('02_phase - FFT Phase Zero Span Plots for ' + zs_phase_str + ' Hz freqs', data_list,
                               name_list, plot_on, x_sync, tag_arr, channel, results_path)
        # os.system("taskkill /im chrome.exe /f")
