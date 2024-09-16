# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 11:26:26 2020

@author: Noam.D
"""

# <codecell> Improts
import sys
# insert at 1, 0 is the script path (or '' in REPL)
# sys.path.insert(1, 'C:/IlyaG_Spyder/Costum Functions')
import os
import time
import pandas as pd
import igf
import igf2
import gc
import ndf
import pywt
# import plotly as py
# import plotly.graph_objs as go
import numpy as np
import plotly.express as px
import plotly as py
import plotly.graph_objs as go
import math
import statistics 
import scipy.fftpack
from scipy  import signal
import matplotlib.pyplot as plt
import webbrowser



W  = '\033[0m'  # white (normal)
R  = '\033[31m' # red
G  = '\033[32m' # green
O  = '\033[33m' # orange
B  = '\033[34m' # blue
P  = '\033[35m' # purple


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

f=25e3
il=1
vc=20e-3
Vout=2
Vin=20


D=Vout/Vin
L=Vin*(D*(1-D))/(f*il)
C=Vin*(D*(1-D))/(8*L*vc*f*f)

print('L is ' +str(L*10**6))
print('C is '+ str(C*10**6))






# <codecell> Directory, Files, Tags, Data location

read_all_folder=True


lab_folder=r'M:\Users\Eddy A\Scripts\Python - Arc Analysis'

#lab_folder='LV450 ARC Measurements - OLD'
path=lab_folder+'\\'


if not os.path.exists(lab_folder):
    os.makedirs(lab_folder)
test_name = 'Arc Analysis'
file_name_arr=[]
file_name_a1=[]
# file_name_arr = ['arc 1 SPI','arc 2 SPI','arc 3 SPI','clean no telem 1 SPI','clean no telem 2 SPI','rst1 SPI','rst2 SPI','rst3 SPI','telem1 SPI','telem2 SPI','telem3 SPI','slow arc 0.25 2 SPI','slow arc 0.25 SPI']
# tag_arr =  ['arc 1 SPI','arc 2 SPI','arc 3 SPI','clean no telem 1 SPI','clean no telem 2 SPI','rst1 SPI','rst2 SPI','rst3 SPI','telem1 SPI','telem2 SPI','telem3 SPI','slow arc 0.25 2 SPI','slow arc 0.25 SPI']






file_neme_search='mana.csv'
if read_all_folder:
    file_arr_temp=[]
    file_name_arr += [each for each in os.listdir(lab_folder+'\\Records 2') if each.endswith(file_neme_search)]
    for filename in file_name_arr:
      print(filename)
      file_arr_temp+=[os.path.splitext(filename)[0]]
    file_name_arr=[]   
    file_name_arr=  file_arr_temp


# file_name_arr = ['arc 2 SPI']
tag_arr = file_name_arr
import os


spi_ch_arr = ['Vin','Vout','Iin','IL','Iin','IL']
# spi_ch_arr = ['Vin']
# spi_ch_arr = ['Vin','Vout','Iin','IL']
results_path=path+test_name
# spi_ch_arr = ['Rx']
if not os.path.exists(results_path):
    os.makedirs(results_path)

print('\nStart Run...')  






# <codecell> Read Data Frame from SCOPE or SPI Data CSV file
filename_counter=0
scope_ch_arr=["Varc_DC", "Varc_AC", "Iarc_DC", "Iarc_AC"]
record_folder = path + 'Records 2' + '\\'
for filename in file_name_arr:
    results_path=path+test_name+'\\'+  filename
    print(R+ filename +W)
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    scope_file_name_arr = [s for s in file_name_arr if "SCOPE" in s]
    scope_tag_arr = [s for s in tag_arr if "SCOPE" in s]
        
    spi_file_name_arr = [s for s in file_name_arr if "SPI" in s]
    spi_tag_arr = [s for s in tag_arr if "SPI" in s]
    if len(scope_file_name_arr)>0:
        ch_arr=scope_ch_arr
        ch_arr=scope_file_name_arr
        df=igf.scope_CSV_to_df(record_folder, scope_file_name_arr, scope_tag_arr, scope_ch_arr)    
    
    if len(spi_file_name_arr)>0:
        spi_Fs=133e3
        spi_filename=filename
        ch_arr=spi_ch_arr
        df = igf.spi_TXT_to_df(path, spi_filename, spi_tag_arr[filename_counter], spi_ch_arr, spi_Fs)
        print(df.head)
    filename_counter+=1


    df.to_csv(results_path +' df.csv')
# df2=df
# df2=df2.drop('Time', axis=1, inplace=False)
# df2['Ratio']=df2.iloc[:,1:2].values/df2.iloc[:,0:1].values ##Vout/Vin
    # <codecell> DataFrame Chunk
    t_start =0
    t_end = 40
    df1 = igf.df_Chunk(df, t_start, t_end)
    df=df1
    gc.collect()
    
    var_win_factor=100
    avg_win_factor=20
    
    # <codecell> DataFrame Simple Down Sample
    down_sample = 1
    df1 = df1.iloc[::down_sample]
    Fs = round(1/(df1['Time'].iloc[1]-df1['Time'].iloc[0]))     # Data samples frequency
    print(Fs)
    # <codecell> Reset Chunk Time
    df1 = igf.df_time_reset(df1,'Time')
    tmax=max(df1['Time'])
    # <codecell> Variance and std Calculations on RAW data
    var_win_t = 0.05
    # if var_win_t>(tmax/var_win_factor):
    #     var_win_t = tmax/var_win_factor
    win_size = int(var_win_t*Fs)
    df_var = igf.df_var_calc(df1, win_size, 'Time')
    df_var_dB = igf.df_dB_calc(df_var, 'Time')
    
    #%% Calculate of Variance and std averaging
    avg_win_t = 0.1
    # if avg_win_t>(tmax/avg_win_factor):
    #     avg_win_t = tmax/avg_win_factor
        
    win_size = int(avg_win_t*Fs)
    df_var_mean = igf.df_mean_calc(df_var, win_size, 'Time')
    
    df_var_mean_dB = igf.df_dB_calc(df_var_mean, 'Time')
    new_col=df_var_mean_dB.columns[1:]
    df_var_dB[new_col]=df_var_mean_dB[new_col]
    
    del df_var
    del df_var_mean
    del df_var_mean_dB
    gc.collect()
    #%% Plots - SCOPE RAW, Variance, Variance mean, Variance dB
    max_plot_res=100000
    for channel in ch_arr:
        plot_on=False
        file_on=False
        
        scope_plot_data = igf.data_plot(df1, 'RAW Data', 'Time', max_plot_res, channel, plot_on, file_on, results_path)
        var_plot_data = igf.data_plot(df_var_dB, 'Variance', 'Time', max_plot_res, channel, plot_on, file_on, results_path)
      
        plot_on=True
        data_list=[scope_plot_data, var_plot_data]
        name_list=['Scope RAW', 'Scope Variance [dB]']
        x_sync=True
        data_name = '01 - Time Domain RAW Data Stats [RAW-Variance-Mean]'
        igf.data_pane_plot(data_name, data_list, name_list, plot_on, x_sync, tag_arr, channel, results_path)
    
    del df_var_dB
    gc.collect()
    os.system("taskkill /im chrome.exe /f")
    # <codecell> STFT Spectrogram Calculation of DataFrame

    f_resolution = 100     # Spectrum resolution RBW
    t_resolution = 0.001   # Time resolution
    
    fft_win = int(Fs/f_resolution)
    fft_win_overlap = int(fft_win-(t_resolution*Fs))
    # fft_win = 10000
    # fft_win_overlap = 5000
    
    if fft_win>len(df1):
        
        fft_win = int(len(df1)/10)
        fft_win_overlap = int(fft_win*0.99)
    res_name_arr=[]
    (t, f, Zraw, res_name_arr) = igf.df_stft_RAW_calc(df1, Fs, fft_win, fft_win_overlap, ch_arr)
    # f=f[f<200000]
    calc_mag = True
    calc_phase = True
    
    
    # <codecell> Z magnitude calculations
    calc_mag=True
    if calc_mag:
        ZdBm = igf.Z_mag_calc(Zraw)
    
    # <codecell> Z phase calculations
    if calc_phase:
        phase_unwrap=False
        Zphase = igf.Z_phase_calc(Zraw, phase_unwrap)
    
    # <codecell> Plot - Spectrogram
    spectrogram_on=True
    if spectrogram_on:
        fmin=0
        fmax=150000
        t_start=0
        max_plot_res=5000
        plot_on=True
        for i in range(len(Zraw)):
            meas_sig=res_name_arr[i]
            print( O+meas_sig+W )
            if calc_mag:
                igf.scpectrogram_plot(ZdBm[i], t, f, max_plot_res, fmax, fmin, t_start, plot_on, results_path, meas_sig+'_amp')
            if calc_phase:
                igf.scpectrogram_plot(Zphase[i], t, f, max_plot_res, fmax, fmin, t_start, plot_on, results_path, meas_sig+'_phase')
                os.system("taskkill /im chrome.exe /f")
    # <codecell> ZeroSpan results for STFT results
    save_zero_span=True
    calc_phase=True
    if calc_mag:
        # zero_span_arr=[53e3, 106e3, 159e3]
        zero_span_arr=[3e3,4e3,5.2e3,15e3,16.6e3,20e3,25e3]
        df_fft = igf.ZeroSpan_calc(ZdBm, res_name_arr, t, f, zero_span_arr, ch_arr)
        zs_amp_str = str(zero_span_arr).replace(",", " -")
        ndf.Save_df_fft_mag(results_path,df_fft,filename)
    
    if calc_phase:
        zero_span_phase_arr=[3e3,4e3,5.2e3,15e3,16.6e3,20e3,25e3]
        df_fft_phase = igf.ZeroSpan_calc(Zphase, res_name_arr, t, f, zero_span_phase_arr, ch_arr)
        cols=df_fft_phase.columns[1:]
        df_fft_phase[cols] = df_fft_phase[cols].diff()
        zs_phase_str = str(zero_span_phase_arr).replace(",", " -")
        ndf.Save_df_fft_phase(results_path,df_fft_phase,filename)
    
    
    
    # <codecell> Plot - Scope RAW, Zero Span FFT, Variance of Zero Span FFT, Mean of Zero Span FFT
    max_plot_res=100000
    calc_mag=True
    if calc_mag:
        # for channel in ch_arr:
        var_win_t = 0.05
        win_size = int(var_win_t/t_resolution)
        fft_var = igf.df_var_calc(df_fft, win_size, 't')
        
        win_size = int(avg_win_t/t_resolution)
        fft_var_mean = igf.df_mean_calc(fft_var, win_size, 't')
        new_col=fft_var_mean.columns[1:]
        fft_var[new_col]=fft_var_mean[new_col]
            
        fft_mean = igf.df_mean_calc(df_fft, win_size, 't')
        new_col=fft_mean.columns[1:]
        df_fft[new_col]=fft_mean[new_col]
        
        del fft_var_mean
        del fft_mean
        gc.collect()
        
        for channel in ch_arr:        
            file_on=False
            plot_on=False
            scope_plot_data = igf.data_plot(df1, 'SPI RAW Data', 'Time', max_plot_res, channel, plot_on, file_on, results_path)
            fft_plot_data = igf.data_plot(df_fft, 'Zero Span FFT', 't', max_plot_res, channel, plot_on, file_on, results_path)
            fft_var_plot_data = igf.data_plot(fft_var, 'Zero Span FFT Variance', 't', max_plot_res, channel, plot_on, file_on, results_path)
           
            plot_on=True
            data_list=[scope_plot_data, fft_plot_data]
            name_list=['RAW Data', 'ZeroSpan FFT']
            data_list=[scope_plot_data, fft_plot_data, fft_var_plot_data]
            name_list=['RAW Data', 'ZeroSpan FFT', 'ZeroSpan FFT Variance']
            x_sync=True
            igf.data_pane_plot('02_mag - FFT Magnitude Zero Span Plots for '+zs_amp_str+' Hz freqs', data_list, name_list, plot_on, x_sync, tag_arr, channel, results_path)
    
    max_plot_res=100000
    if calc_phase:
        # for channel in ch_arr:
        var_win_t = 0.05
        win_size = int(var_win_t/t_resolution)
        fft_var = igf.df_var_calc(df_fft_phase, win_size, 't')
        
        avg_win_t = 0.1
        win_size = int(avg_win_t/t_resolution)
        fft_var_mean = igf.df_mean_calc(fft_var, win_size, 't')
        new_col=fft_var_mean.columns[1:]
        fft_var[new_col]=fft_var_mean[new_col]
            
        fft_mean = igf.df_mean_calc(df_fft_phase, win_size, 't')
        new_col=fft_mean.columns[1:]
        df_fft_phase[new_col]=fft_mean[new_col]
        
        del fft_var_mean
        del fft_mean
        gc.collect()
        for channel in ch_arr:        
            file_on=False
            plot_on=False
            scope_plot_data = igf.data_plot(df1, 'Scope RAW Data', 'Time', max_plot_res, channel, plot_on, file_on, results_path)
            fft_phase_plot_data = igf.data_plot(df_fft_phase, 'Zero Span FFT', 't', max_plot_res, channel, plot_on, file_on, results_path)
            fft_var_plot_data = igf.data_plot(fft_var, 'Zero Span FFT Variance', 't', max_plot_res, channel, plot_on, file_on, results_path)
           
            plot_on=True
            data_list=[scope_plot_data, fft_phase_plot_data, fft_var_plot_data]
            name_list=['RAW Data', 'ZeroSpan Phase', 'ZeroSpan Phase Variance']
            x_sync=True
            igf.data_pane_plot('02_phase - FFT Phase Zero Span Plots for '+zs_phase_str+' Hz freqs', data_list, name_list, plot_on, x_sync, tag_arr, channel, results_path)
        os.system("taskkill /im chrome.exe /f")
    
    # <codecell> Sliding Window Magnitude Spectrum Analysis and Plots
    if calc_mag:
        MH_time = 0.1
        Overlap_time = 0.05
        
        win_size=int(MH_time/t_resolution)
        win_overlap=int(Overlap_time/t_resolution)
        max_plot_res=100000
        
        for channel in ch_arr:
            df_MH_1 = pd.DataFrame(columns=['f'])
            df_AVG_1 = pd.DataFrame(columns=['f'])
            
            df_MH_1['f']=f
            df_AVG_1['f']=f
            
            indices = [i for i, elem in enumerate(res_name_arr) if channel in elem]
            
            for i in indices:
                meas_sig=res_name_arr[i]
                if channel in meas_sig:
                    (df_MH_temp, df_AVG_temp,t_list) = igf.sliding_spectrum(ZdBm[i], t, f, win_size, win_overlap, meas_sig)
            
                df_MH_1[df_MH_temp.columns]=df_MH_temp
                df_AVG_1[df_AVG_temp.columns]=df_AVG_temp
            
            del df_MH_temp
            del df_AVG_temp
            file_on=False
            plot_on=False
            scope_plot_data = igf.data_plot(df1, 'Scope RAW Data', 'Time', max_plot_res, channel, plot_on, file_on, results_path)
            MH1_plot_data = igf.data_plot(df_MH_1, 'Sliding Spectrum MH', 'f', max_plot_res, channel, plot_on, file_on, results_path)
            AVG1_plot_data = igf.data_plot(df_AVG_1, 'Sliding Spectrum AVG', 'f', max_plot_res, channel, plot_on, file_on, results_path)
            
            plot_on=False
            data_list=[scope_plot_data, MH1_plot_data]
            name_list=['ZeroSpan Results', 'Sliding MH Spectrum dBm']
            # data_list=[ MH1_plot_data]
            # name_list=['Sliding MH Spectrum dBm']
            x_sync=False
            data_name = '03_mag - Sliding FFT MH Spectrum with ZeroSpan'
            igf.data_pane_plot(data_name, data_list, name_list, plot_on, x_sync, tag_arr, channel, results_path)
            
            plot_on=False
            data_list=[scope_plot_data, AVG1_plot_data]
            name_list=['ZeroSpan Results', 'Sliding AVG Spectrum dBm']
            # data_list=[ AVG1_plot_data]
            # name_list=['Sliding AVG Spectrum dBm']
            x_sync=False
            data_name = '04_mag - Sliding FFT AVG Spectrum with ZeroSpan'
            igf.data_pane_plot(data_name, data_list, name_list, plot_on, x_sync, tag_arr, channel, results_path)        
            
            plot_on=True
            data_list=[MH1_plot_data, AVG1_plot_data]
            name_list=['Sliding MH Spectrum dBm', 'Sliding AVG Spectrum dBm']
            x_sync=True
            data_name = '05_mag - Sliding FFT MH and AVG Spectrum'
            igf.data_pane_plot(data_name, data_list, name_list, plot_on, x_sync, tag_arr, channel, results_path)
        os.system("taskkill /im chrome.exe /f")
    # <codecell> Sliding Window Phase Spectrum Analysis and Plots
    slide_FFT_phase=True;
    if calc_phase and slide_FFT_phase:
        MH_time = 0.1
        Overlap_time = 0.05
        
        win_size=int(MH_time/t_resolution)
        win_overlap=int(Overlap_time/t_resolution)
        max_plot_res=100000
        
        for channel in ch_arr:
            df_MH_1 = pd.DataFrame(columns=['f'])
            df_AVG_1 = pd.DataFrame(columns=['f'])
            
            df_MH_1['f']=f
            df_AVG_1['f']=f
            
            indices = [i for i, elem in enumerate(res_name_arr) if channel in elem]
            
            for i in indices:
                meas_sig=res_name_arr[i]
                if channel in meas_sig:
                    (df_MH_temp, df_AVG_temp,t_list) = igf.sliding_spectrum(Zphase[i], t, f, win_size, win_overlap, meas_sig)
            
                df_MH_1[df_MH_temp.columns]=df_MH_temp
                df_AVG_1[df_AVG_temp.columns]=df_AVG_temp
            
            del df_MH_temp
            del df_AVG_temp
            file_on=False
            plot_on=False
            scope_plot_data = igf.data_plot(df1, 'Scope RAW Data', 'Time', max_plot_res, channel, plot_on, file_on, results_path)
            MH1_plot_data = igf.data_plot(df_MH_1, 'Sliding Spectrum MH', 'f', max_plot_res, channel, plot_on, file_on, results_path)
            AVG1_plot_data = igf.data_plot(df_AVG_1, 'Sliding Spectrum AVG', 'f', max_plot_res, channel, plot_on, file_on, results_path)
            
            plot_on=False
            data_list=[scope_plot_data, MH1_plot_data]
            name_list=['Scope RAW', 'Sliding MH Spectrum dBm']
            x_sync=False
            data_name = '03_phase - Sliding FFT MH Spectrum with Scope RAW'
            igf.data_pane_plot(data_name, data_list, name_list, plot_on, x_sync, tag_arr, channel, results_path)
    
            plot_on=False
            data_list=[scope_plot_data, AVG1_plot_data]
            name_list=['Scope RAW', 'Sliding AVG Spectrum dBm']
            x_sync=False
            data_name = '04_phase - Sliding FFT AVG Spectrum with Scope RAW'
            igf.data_pane_plot(data_name, data_list, name_list, plot_on, x_sync, tag_arr, channel, results_path)
            
            plot_on=True
            data_list=[MH1_plot_data, AVG1_plot_data]
            name_list=['Sliding MH Spectrum dBm', 'Sliding AVG Spectrum dBm']
            x_sync=True
            data_name = '05_phase - Sliding FFT MH and AVG Spectrum'
            igf.data_pane_plot(data_name, data_list, name_list, plot_on, x_sync, tag_arr, channel, results_path)
            
    
    os.system("taskkill /im chrome.exe /f")
     # <codecell> End Sequence       
    f_resolution = 100   # Spectrum resolution RBW
    t_resolution = 0.05  # Time resolution
      
       
    fft_win = int(Fs/f_resolution)
    fft_win_overlap = int(fft_win-(t_resolution*Fs))
    plot_on=True
    waveletname = 'db9'    
    Maximum_decomposition_level=pywt.dwt_max_level(len(df), waveletname)
    print("Maximum_decomposition_level is %s " %(Maximum_decomposition_level))
    for col in df1.iloc[:,1:]:
        df_wavelet=igf2.Sliding_WDT(df1,'Time',col,fft_win,fft_win_overlap,waveletname,6)
        res = []
        
        res = []
        for  wavelet_col in df_wavelet.columns[2:]:
            res.append(
                go.Bar(
                    x=df_wavelet['Time'],
                    y=df_wavelet[col],
                    name=wavelet_col
                )
            )
        layout = go.Layout(
            barmode='group'
        )
        fig = go.Figure(data=res, layout=layout)
        config = {'scrollZoom': True}
        # fig['layout'].update(title=string+ 'SNR' )
        fig['layout'].update(xaxis=dict(title = 'time'))
        fig['layout'].update(yaxis=dict(title = 'coeff power RMS'))
        # data_out.append(data)
        results_path=results_path+'/'
        if not os.path.exists(results_path):
            os.makedirs(results_path)
        print("Generated plot file")
        py.offline.plot(fig,auto_open = plot_on, config=config, filename=results_path+'/'+col+' df_wavelet.html')    
        
          
                     
        print('\nStop Run')
        tend = time.time()
        print('---| Runtime = '+str(tend-tstart)+' Sec') 
    # df_wavelet2=igf2.DF_to_WDT_DF(df,'Time','Rec034_SPI_Vin',waveletname,5)
     # <codecell> End Sequence      

# import pywt
# t_start =0
# t_end = 17.5

# df2 = igf.df_Chunk(df, t_start, t_end)
# Sig_arr=df2.iloc[:,1:2].to_numpy().flatten()
# chirp_signal=df2.iloc[:,1:2].to_numpy().flatten()
# # fig, ax = plt.subplots(figsize=(6,1))
# # ax.set_title("Original Chirp Signal: ")
# # ax.plot(chirp_signal)
# # plt.show()
    
# data = chirp_signal
# waveletname = 'sym5'
# waveletname = 'db9'  
# cA_list= []
# cD_list= []
# fig, axarr = plt.subplots(nrows=6, ncols=2, figsize=(6,6))
# Maximum_decomposition_level=pywt.dwt_max_level(len(df2), waveletname)
# for ii in range(6):
#     (data, coeff_d) = pywt.dwt(data, waveletname)
#     axarr[ii, 0].plot(data, 'r')
#     axarr[ii, 1].plot(coeff_d, 'g')
#     axarr[ii, 0].set_ylabel("Level {}".format(ii + 1), fontsize=14, rotation=90)
#     axarr[ii, 0].set_yticklabels([])


#     cD_list.append(coeff_d)
#     cA_list.append(data)
#     if ii == 0:
#         axarr[ii, 0].set_title("Approximation coefficients", fontsize=14)
#         axarr[ii, 1].set_title("Detail coefficients", fontsize=14)
#     axarr[ii, 1].set_yticklabels([])
# plt.tight_layout()
# plt.show()


# coeff=[]
# coeff.insert(0,cA_list[len(cA_list)-1])
# coeff=coeff+cD_list[::-1]

# y=pywt.waverec(coeff,waveletname)

     # <codecell> End Sequence 

# max_plot_res=100e3
# plot_on=False
# file_on=True
# plot_on=True

# wavelet_plot_data_reconstructed = ndf.ndf_data_plot(df_wavelet, 'reconstarced Data', 'Time', max_plot_res, plot_on, file_on, results_path)


# x = [1,2,3]
# y = [[1,2,3],[4,5,6],[7,8,9]]
# plt.xlabel("X-axis")
# plt.ylabel("Y-axis")
# plt.title("A test graph")
# for i in range(len(y[0])):
#     plt.plot(x,[pt[i] for pt in y],label = 'id %s'%i)
# plt.legend()
# plt.show()

     # <codecell> End Sequence 

# import plotly.express as px
# import plotly as py
# import plotly.graph_objs as go
# max_plot_res=100000
# plot_res=int(len(df_wavelet)/max_plot_res)
# if plot_res==0:
#     plot_res=1
    
    
# # dfx=df_wavelet.iloc[::plot_res]
# x_data = df_wavelet2['Time']




# # fig = px.scatter(df_wavelet.iloc[1:], x='Time',y=y_data)
# # config = {'scrollZoom': True}
# # wdt_txt=' - '+' Classification'
# # fig['layout'].update(title=wdt_txt)
# #     # fig.update(layout_showlegend=False)
# # txt=results_path+'/'+wdt_txt
# # py.offline.plot(fig,auto_open = True, config=config, filename=txt+'.html')

# res = []
# for col in df_wavelet2.columns[2:]:
#     res.append(
#         go.Bar(
#             x=df_wavelet['Time'],
#             y=df_wavelet[col],
#             name=col
#         )
#     )
# layout = go.Layout(
#     barmode='group'
# )
# fig = go.Figure(data=res, layout=layout)
# config = {'scrollZoom': True}
# # fig['layout'].update(title=string+ 'SNR' )
# fig['layout'].update(xaxis=dict(title = 'time'))
# fig['layout'].update(yaxis=dict(title = 'coeff power RMS'))
# # data_out.append(data)
# results_path=results_path+'/'
# if not os.path.exists(results_path):
#     os.makedirs(results_path)
# print("Generated plot file")
# py.offline.plot(fig,auto_open = plot, config=config, filename=results_path+'/'+filename+' df_wavelet.html')    
