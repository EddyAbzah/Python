"""
    Function module which includes all relevant signal processing
    
    @author: Noam dahan
    @last update: 25/07/2020
"""

# <codecell> Imports
import pandas as pd
import plotly as py
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import numpy as np
# import scipy.fftpack
import math
from scipy import signal
import time
import gc
import matplotlib.pyplot as plt
import statistics 
import os
import numpy as np 
from scipy  import signal
from statistics import mean 
from art import *
import igf

W  = '\033[0m'  # white (normal)
R  = '\033[31m' # red
G  = '\033[32m' # green
O  = '\033[33m' # orange
B  = '\033[34m' # blue
P  = '\033[35m' # purple

color = {
    'white':    "\033[1;37m",
    'yellow':   "\033[1;33m",
    'green':    "\033[1;32m",
    'blue':     "\033[1;34m",
    'cyan':     "\033[1;36m",
    'red':      "\033[1;31m",
    'magenta':  "\033[1;35m",
    
    'black':      "\033[1;30m",
    'darkwhite':  "\033[0;37m",
    'darkyellow': "\033[0;33m",
    'darkgreen':  "\033[0;32m",
    'darkblue':   "\033[0;34m",
    'darkcyan':   "\033[0;36m",
    'darkred':    "\033[0;31m",
    'darkmagenta':"\033[0;35m",
    'darkblack':  "\033[0;30m",
    'off':        "\033[0;0m"
}

tprint("start run...",font="rnd-medium")
# <codecell> class

class Arc:

    def __init__(self, Time):
        self.Arc_Time = Time
        # creates a new type for each Arc


class Telemerty:

    def __init__(self, Time):
        self.Telemerty_Time = Time
        # creates a new type for each Telem



a=Arc(1.5)


# <codecell> Read npz file (Zraw,f,t)
def npz_to_array(path):
    
    data = np.load(path+'/mat.npz')
   
    return data.f.t ,data.f.f, data.f.Zraw


def Save_Zraw(results_path,t,f,Zraw):
    
    np.savez(results_path+'\mat.npz', t=t, f=f,Zraw=Zraw)
    return
   
    
def Save_df_fft_mag(results_path,df_fft,test_name):
    df_fft.to_csv(results_path+'/'+ test_name +' df_fft_mag.csv', index = False, header=True)
    
def Save_df_fft_phase(results_path,df_fft,test_name):
    df_fft.to_csv(results_path+'/'+ test_name +' df_fft_phase.csv', index = False, header=True)
def df_dBTOv_calc_all(dfi):
    """
        Function calculates 20*Log10(dfi) dataframe columns
        `dfi`       - input Pandas Data Frame
        `x_col`     - exact x axis col name

    """
    start = time.time()

    dfi_V=pd.DataFrame()
    col=dfi.columns[1:]
    #dfi[dfi==240]=10**-12
    dfi_V[col]=np.exp(dfi[col]/20)
    # dfi_V=dfi_V.rename(columns={dfi_V.columns[0]: x_col})
    end = time.time()
    print('--| Runtime = '+str(end-start)+' Sec')
    
    return(dfi_V)    
           
def df_dBTOv_calc(dfi, x_col):
    """
        Function calculates 20*Log10(dfi) dataframe columns
        `dfi`       - input Pandas Data Frame
        `x_col`     - exact x axis col name

    """
    start = time.time()

    dfi_V=pd.DataFrame()
    dfi_V[x_col]=dfi[x_col]
    col=dfi.columns[1:]
    #dfi[dfi==240]=10**-12
    dfi_V[col]=np.exp(dfi[col]/20)
    dfi_V=dfi_V.rename(columns={dfi_V.columns[0]: x_col})
    end = time.time()
    print('--| Runtime = '+str(end-start)+' Sec')
    
    return(dfi_V) 
def Read_df_fft(path):
    df_fft= pd.read_csv(path, header=0) # Reading the df_fft from the  LV450 Basic AFD Evaluation V4.0
    df_fft = df_fft[df_fft['t']>0.9] #removing junk
    return df_fft


def SNR(df_fft,zero_span_arr,results_path,plot,factor,string):
    df_fft = df_fft[df_fft['t']>0.9] 
    t=df_fft[['t']]
    win_size=10
    data=[]
    SNR= pd.DataFrame()
    for col in df_fft.columns[1:]:
        print(col)
        # plt.plot(t,df_fft[col])
        # plt.show()
        Signal=df_dBTOv_calc(df_fft,col)# in VOLTS
        
        Noise=Signal[col].rolling(window=win_size).std().median()/np.sqrt(win_size)  
        #print(Noise)
        AVG=Signal[col].rolling(window=win_size).mean().median()
        #print(AVG)
        TEMP=(abs(Signal[col]-AVG))/Noise
        TEMP[TEMP==0]=10**-12
        SNR[col]=20*np.log10(TEMP)
        
        
        sp = np.fft.fft(np.sin(t))
        freq = np.fft.fftfreq(t.shape[-1])
                
        data.append( 
                    go.Scattergl(
                    x=t['t'],
                    y=SNR[col],
                    mode = 'lines',
                    name= col,
                    #mode = 'markers',
                    hoverlabel = dict( namelength = -1) 
                    )
                )
        Signal=[]
        Noise=0
        AVG=0
 
    fig = go.FigureWidget(data)
    config = {'scrollZoom': True}
    fig['layout'].update(title=string+ 'SNR' )
    fig['layout'].update(xaxis=dict(title = 'time'))
    fig['layout'].update(yaxis=dict(title = 'SNR[dB]'))
    # data_out.append(data)
    results_path=results_path+'/'+str(factor)
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    print("Generated plot file")
    py.offline.plot(fig,auto_open = plot, config=config, filename=results_path+'/'+string+' SNR ds by '+str(factor)+', the df_fft is '+str(round(zero_span_arr[0]/1000,2))+'[Khz].html')    
    return SNR,t['t']

def MH_plot_for_gui(res_name_arr,ZdBm,t,f,MH_time,Overlap_time,name, Factor,ch_arr,plot,results_path):
    
    
   t_resolution = 0.001  
   win_size=int(MH_time/t_resolution)    
   win_overlap=int(Overlap_time/t_resolution)
   max_plot_res=100000 
    
    
    
   indices = [i for i, elem in enumerate(res_name_arr)]
       
   
   df_MH_1 = pd.DataFrame(columns=['f'])
   df_AVG_1 = pd.DataFrame(columns=['f'])
       
   df_MH_1['f']=f
   df_AVG_1['f']=f
   for i in indices:
       print(i)
       meas_sig=res_name_arr[i]
       (df_MH_temp, df_AVG_temp,t_list) = igf.sliding_spectrum(ZdBm[i], t, f, win_size, win_overlap, meas_sig)
 
       df_MH_1[df_MH_temp.columns]=df_MH_temp
       df_AVG_1[df_AVG_temp.columns]=df_AVG_temp
 
       del df_MH_temp
       del df_AVG_temp
   file_on=False
   plot_on=False
   MH1_plot_data = igf.data_plot(df_MH_1, 'Sliding Spectrum MH', 'f', max_plot_res, ch_arr[0], plot_on, file_on, results_path)
      
      
   plot_on=True
   data_list=[MH1_plot_data]
    
   name_list=['Sliding MH Spectrum dBm']
   x_sync=True
   data_name =str(Factor) + name+' Sliding FFT MH'
   igf.data_pane_plot(data_name, data_list, name_list, plot_on, x_sync, ch_arr, ch_arr[0], results_path) 
   
   
   
   
   
   
def MH_plot(res_name_arr,ZdBm,t,f,MH_time,Overlap_time,name, Factor,ch_arr,plot,results_path):
    
    
   t_resolution = 0.001  
   win_size=int(MH_time/t_resolution)    
   win_overlap=int(Overlap_time/t_resolution)
   max_plot_res=100000 
    
    
    
   indices = [i for i, elem in enumerate(res_name_arr)]
       
   
   df_MH_1 = pd.DataFrame(columns=['f'])
   df_AVG_1 = pd.DataFrame(columns=['f'])
       
   df_MH_1['f']=f
   df_AVG_1['f']=f
   for i in indices:
       print(i)
       meas_sig=res_name_arr[i]
       (df_MH_temp, df_AVG_temp,t_list) = igf.sliding_spectrum(ZdBm[i], t, f, win_size, win_overlap, meas_sig)
 
       df_MH_1[df_MH_temp.columns]=df_MH_temp
       df_AVG_1[df_AVG_temp.columns]=df_AVG_temp
 
       del df_MH_temp
       del df_AVG_temp
   file_on=False
   plot_on=False
   MH1_plot_data = igf.data_plot(df_MH_1, 'Sliding Spectrum MH', 'f', max_plot_res, ch_arr[0], plot_on, file_on, results_path)
      
      
   plot_on=True
   data_list=[MH1_plot_data]
    
   name_list=['Sliding MH Spectrum dBm']
   x_sync=True
   data_name =str(Factor) + ' '+ name+ ' Sliding FFT MH and AVG Spectrum'
   igf.data_pane_plot(data_name, data_list, name_list, plot_on, x_sync, tag_arr, ch_arr[0], results_path) 


def SNR_plus(df_fft,zero_span_arr,results_path,plot,factor,string):
    df_fft = df_fft[df_fft['t']>0.9] 
    t=df_fft[['t']]
    win_size=10
    data=[]
    data_out=[t]
    SNR= pd.DataFrame()
    Signal_in_time= pd.DataFrame()
    Signal=df_dBTOv_calc_all(df_fft)
    for col in df_fft.columns[1:]:
        for one_freq in zero_span_arr:
            if str(one_freq) in col:
                print(col)
                # plt.plot(t,df_fft[col])
                # plt.show()
                # Signal=df_dBTOv_calc(df_fft,col)# in VOLTS
                window = signal.gaussian(10,5)
                plt.plot(Signal[col])
                plt.show()
                Signal_in_time[col]=np.convolve(Signal[col],window)
                print(Signal_in_time)
                # Signal=df_dBTOv_calc(df_fft,col)# in VOLTS
                plt.plot(Signal_in_time)
                plt.show()
                
                Noise=Signal_in_time[col].rolling(window=win_size).std().median()/np.sqrt(win_size)  
                #print(Noise)
                AVG=Signal_in_time[col].rolling(window=win_size).mean().median()
                #print(AVG)
                TEMP=((abs(Signal_in_time[col]-AVG))/Noise)
                TEMP[TEMP==0]=10**-12
                SNR[col]=20*np.log10(TEMP)
                
                data.append( 
                            go.Scattergl(
                            x=t['t'],
                            y=SNR[col],
                            mode = 'lines',
                            name= col,
                            #mode = 'markers',
                            hoverlabel = dict( namelength = -1) 
                            )
                        )
                Signal_in_time=[]
                Noise=0
                AVG=0
     
    
    fig = go.FigureWidget(data)
    config = {'scrollZoom': True}
    # fig['layout'].update(title=name)
    # fig['layout'].update(xaxis=dict(title = t))
    data_out.append(data)
    
    print("Generated plot file")
    py.offline.plot(fig,auto_open = True, config=config, filename=results_path+'/'+'df_fft.html')    
    return data_out

def SNR_Matced(df_fft,zero_span_arr,results_path,plot,factor,string):
    df_fft = df_fft[df_fft['t']>0.9] 
    t=df_fft[['t']]
    win_size=10
    data=[]
    data_out=[t]  
    Signal_in_time= pd.DataFrame()
    SNR= pd.DataFrame()
    for col in df_fft.columns[1:]:
        print(col)
        # plt.plot(t,df_fft[col])
        # plt.show()
        Signal=df_dBTOv_calc(df_fft,col)# in VOLTS
        window = signal.gaussian(10,0.05)
        # plt.plot(Signal[col])
        # plt.show()
        Signal_in_time[col]=np.convolve(Signal[col],window)
        Noise=Signal[col].rolling(window=win_size).std().median()/np.sqrt(win_size)  
        #print(Noise)
        AVG=Signal[col].rolling(window=win_size).mean().median()
        #print(AVG)
        TEMP=(abs(Signal_in_time[col]-AVG))/Noise
        TEMP[TEMP==0]=10**-12
        SNR[col]=20*np.log10(TEMP)
        
        
        sp = np.fft.fft(np.sin(t))
        freq = np.fft.fftfreq(t.shape[-1])
                
        data.append( 
                    go.Scattergl(
                    x=t['t'],
                    y=SNR[col],
                    mode = 'lines',
                    name= col,
                    #mode = 'markers',
                    hoverlabel = dict( namelength = -1) 
                    )
                )
        Signal=[]
        Noise=0
        AVG=0
 
    fig = go.FigureWidget(data)
    config = {'scrollZoom': True}
    fig['layout'].update(title=string+ 'SNR' )
    fig['layout'].update(xaxis=dict(title = 'time'))
    fig['layout'].update(yaxis=dict(title = 'SNR[dB]'))
    # data_out.append(data)
    results_path=results_path+'/'+str(factor)
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    print("Generated plot file")
    py.offline.plot(fig,auto_open = plot, config=config, filename=results_path+'/'+string+' SNR ds by '+str(factor)+', the df_fft is '+str(round(zero_span_arr[0]/1000,2))+'[Khz].html')    
    return SNR,t['t']


def Had_telem(df_fft,zero_span_arr,results_path,win_size):
    
    df_fft = df_fft[df_fft['t']>1] 
    t=df_fft[['t']]
    df_xcor = pd.DataFrame(columns=['t'])
    df_xcor['t'] = df_fft['t']
    win_size=10
    data_snr=[]
    data_out_snr=[t]
    data=[]
    data_out=[t]
    SNR= pd.DataFrame()
    DIFF= pd.DataFrame()
    for col in df_fft.columns[1:]:
        for one_freq in zero_span_arr:
            if str(one_freq) in col:
                print('for col ' +col)
                # plt.plot(t,df_fft[col])
                # plt.show()
                Signal=df_dBTOv_calc(df_fft,col)# in VOLTS
                Signal.to_csv(results_path+'/' +' signal.csv', index = False, header=True)
                Noise=Signal[col].rolling(window=win_size).std().median()/np.sqrt(win_size)  
                #print(Noise)
                AVG=Signal[col].rolling(window=win_size).mean().median()
                #print(AVG)
                TEMP=(abs(Signal[col]-AVG))/Noise
                TEMP.to_csv(results_path+'/' +' temp.csv', index = False, header=True)
                TEMP[TEMP==0]=10**-12
                SNR[col]=20*np.log10(TEMP)
                SNR.to_csv(results_path+'/' +' snr.csv')
                AVG_SNR=SNR[col].rolling(window=win_size).mean().median()
                TOP_SNR=AVG_SNR*2
                DIFF=SNR[col]
    
                len_of_sig=(len(DIFF))
                i=0
                while i<len_of_sig:#using the snr defien where is the telem!
                    temp=DIFF[i:i+5].values.tolist()
                    res = all(i < j for i, j in zip(temp, temp[1:])) 
                    if res:
                        # print("well maybe arc")
                        # print(i+901)
                        sample=DIFF.iloc[i]
                        if sample>TOP_SNR:
                            # print(sample)
                            # print("telem or arc")
                            # print (i)
                            is_telem=statistics.mean(DIFF[i:i+200].values.tolist())
                            if is_telem>35:                               # print('this is telem')
                                print('---------------------')
                                print('the stat is '+str(is_telem))
                                print('there is telem in '+ str((i+1001)/1000)+'sec')
                                i=i+300
                                print('telem endes in '+ str((i+1101)/1000)+'sec')
                            
                    i+=1
                print('i='+str(i))
                # Signal=[]
                Noise=0
                AVG=0

    print('win size is'+ str(win_size))
    # print(SNR.rolling(window=win_size).cov())
    # df = pd.DataFrame(Signal)
    SNRK=SNR.rolling(window=500).mean()
   # COV=Signal.rolling(window=win_size).cov().unstack()#.to_csv(results_path+'/' +' COV1.csv')
    COV=Signal.rolling(window=win_size).cov().unstack()#.to_csv(results_path+'/' +' COV1.csv')
     # print(COV)
     # NUMPT_COV=COV.to_numpy()
    # print(NUMPT_COV)
     # for col in COV:
     #     tempt=COV[col]
      
    for col in COV.columns:      
        data.append( 
                     go.Scattergl(
                     x=t['t'],
                     y=COV[col],
                     mode = 'lines',
                     name= col[0]+col[1],
                     #mode = 'markers',
                     hoverlabel = dict( namelength = -1) 
                     )
                 )

    fig = go.FigureWidget(data)
    config = {'scrollZoom': True}
     # fig['layout'].update(title=name)
     # fig['layout'].update(xaxis=dict(title = t))
    data_out.append(data)    
    py.offline.plot(fig,auto_open = True, config=config, filename=results_path+'/'+'COV_fft.html')
 
    # print("Generated plot file")
    # py.offline.plot(fig,auto_open = False, config=config, filename=results_path+'/'+'df_fft.html')    
    return data_out ,COV



def stage_1_energy_raise(EnergyDB,WindowSize,FilterSize,OverThresholdLimit):
    EnergyThresholdList=[]
    EnergyThresholdList.append(0)
    SampleIndex=WindowSize + FilterSize + 20# Skipping the first 20 samples because of Inverter noises.
    while (SampleIndex < len(EnergyDB)):
        EnergyThreshold = 4;
        MinFilterWindow=min(EnergyDB[(SampleIndex - WindowSize - FilterSize):SampleIndex - WindowSize])
        while EnergyThreshold<50:
            OverThresholdCounter = 0
            i=SampleIndex - WindowSize 
            while(i<SampleIndex):
                
                if EnergyDB.iloc[i]>(MinFilterWindow+EnergyThreshold+1): 
                # if MinFilterWindow-EnergyDB.iloc[i]>(EnergyThreshold+1): 
                    OverThresholdCounter+=1
                if (OverThresholdCounter >= OverThresholdLimit):
                    break
                i+=1
            if (OverThresholdCounter < OverThresholdLimit):
                break
            EnergyThreshold+=0.5
        if EnergyThreshold == 4:
            EnergyThreshold = 0
        else: 
            EnergyThresholdList.append(EnergyThreshold) 
        SampleIndex+=1   
    print(W+'The max energy raise is '+ P +str(max(EnergyThresholdList)))
    return max(EnergyThresholdList)

def stage_1_Iac_raise(Iac_arr,WindowSize,FilterSize,OverThresholdLimit):
    Iac_ThresholdList=[]
    Iac_ThresholdList.append(0)
    SampleIndex=WindowSize + FilterSize + 20# Skipping the first 20 samples because of Inverter noises.
    while (SampleIndex < len(Iac_arr)):
        MaxCurrentDrop = 0.1;
        MinFilterWindow=(Iac_arr[(SampleIndex - WindowSize - FilterSize):SampleIndex - WindowSize]).mean().iloc[0]
        while MaxCurrentDrop<1:
            OverThresholdCounter = 0
            i=SampleIndex - WindowSize 
            while(i<SampleIndex):
                
                if MinFilterWindow - Iac_arr.iloc[i].values[0]>(+MaxCurrentDrop): 
                # if MinFilterWindow-Iac_arr.iloc[i]>(EnergyThreshold+1): 
                    OverThresholdCounter+=1
                if (OverThresholdCounter >= OverThresholdLimit):
                    break
                i+=1
            if (OverThresholdCounter < OverThresholdLimit):
                break
            MaxCurrentDrop+=0.05
        if MaxCurrentDrop == 0.1:
            MaxCurrentDrop = 0
        else: 
            Iac_ThresholdList.append(MaxCurrentDrop) 
        SampleIndex+=1   
    print(W+'The max jittring in the Iac is '+ R+str(max(Iac_ThresholdList)))
    return max(Iac_ThresholdList)


def cheak_harmonics(f_resampling,fpeak,f_fft,k,good_k):
    for i in range(1,k+1):
        if (abs(f_fft+1000)< abs(f_resampling-i*fpeak) or  abs(f_resampling-i*fpeak)  < abs(f_fft-1000)):
            good_k+=1
    return good_k
            

def Get_downsampled_signal(x, fs, target_fs,order,Lpf_type):
    decimation_ratio = np.round(fs / target_fs)
    if fs < target_fs:
        raise ValueError("Get_downsampled_signal")
    else:
        try:
            y0 = signal.decimate(x,int(decimation_ratio), order,zero_phase=True,ftype=Lpf_type)
            print(Lpf_type)
            print(R+ str(int(decimation_ratio)) +W)
            # y1 = signal.decimate(y0,2, 2,zero_phase=True,ftype='iir')
            # f_poly = signal.resample_poly(y, 100, 20)
        except:
            y0 = signal.decimate(x, int(decimation_ratio), 3)
        actual_fs = fs / decimation_ratio
    return y0, actual_fs 



def Get_downsampled_signal_NO_FILTER(x, fs, target_fs):
    decimation_ratio = np.round(fs / target_fs)
    if fs < target_fs:
        raise ValueError("Get_downsampled_signal")
    else:
        try:
            y0=x[::int(decimation_ratio)]
            print(R+ str(int(decimation_ratio)) +W)
            # y1 = signal.decimate(y0,2, 2,zero_phase=True,ftype='iir')
            # f_poly = signal.resample_poly(y, 100, 20)
        except:
            y0=x[::int(decimation_ratio)]
        actual_fs = fs / decimation_ratio
    return y0, actual_fs 



def poly_resample(psg, new_sample_rate, old_sample_rate):
    return signal.resample_poly(psg, new_sample_rate, old_sample_rate, axis=0)    
            



def spi_TXT_to_df_for_gui(inputFile, tag_arr, spi_sample_rate):
    """
        Function import SPI data TXT file into Pandas DataFrame adding 'Time' vector and calculates Voltage
        
        Inputs:
        `path`              - Data files path (relatively to .py file location); String 
        `filename`          - file name you want to analyse; String
        `tag_arr`           - Array of tags you want to attach to data files; String array 
        `spi_param_arr`     - Array of measured SPI parameters you want to analyse; String array
        `spi_sample_rate`   - SPI sampled data rate [Hz]; integer
        
        Returns:
            DF - Pandas Data Frame
        
        Example of usage : 
            df_spi = spi_TXT_to_df(path, filename, tag_arr, spi_param_arr, spi_sample_rate)
    """
    print('--> Reading SPI Data txt...')
    tsss = time.time()
    Ts=1/spi_sample_rate
    Fs=spi_sample_rate
    df_spi = pd.DataFrame(columns=['Time'])
    df = pd.read_csv(inputFile, header=0)
    i=0
    for col in df.columns:
        if 'Unnamed' in col:
            # print(col)
            del df[col]
        else:
            df = df.rename(columns = {col: tag_arr[i]})
            i=i+1
    
    df=df.add_prefix(tag_arr[0]+'_')
    df_spi['Time'] = (df.index)*Ts
    V_quantization=1/(2**12)
    df_spi[df.columns]=df*V_quantization

    df_len=len(df_spi)
    df_time_len = max(df_spi['Time'])-min(df_spi['Time'])
    tmin=min(df_spi['Time'])
    tmax=max(df_spi['Time'])
      
    temp1='DF Tmin = '+str(tmin)+'[Sec]; '+'DF Tmax = '+str(tmax)+'[Sec]; \n'
    temp2='DF time length = '+str(round(df_time_len,5))+'[Sec] / ~'+str(round(df_time_len/60,4))+'[Min]; \n'
    text=temp1+temp2+'DF length = '+str(df_len/1000000)+'[Mega Samples];\n'+'DF Sampling rate = '+str(round((Fs/1000),0))+'[kSamp/sec]'+'; DF Sampling Interval = '+str(round((Ts*1000),3))+'[mSec]'
    
    print(text)
    print('Finished Reading Data')
    teee = time.time()
    print('--| Runtime = '+str(teee-tsss)+' Sec\n')
    
    return (df_spi)


         
# <codecell> For cheaking the function
    


# lab_folder='C:\projects\Roof Session'
# #lab_folder='LV450 ARC Measurements - OLD'
# path=lab_folder+'\\'

# test_name = 'rec034_results'

# # file_name_arr = ['rec008_ARC_SCOPE', 'rec009_ARC_SCOPE', 'rec010_ARC_SCOPE']
# # tag_arr = ['ARC_008', 'ARC_009', 'ARC_010']

# file_name_arr = ['Rec034_SPI']
# tag_arr = ['ARC_034_SPI']

# # file_name_arr = ['Rec003_SCOPE']
# # tag_arr = ['Rec003_SCOPE']

# # scope_ch_arr = ['Varc', 'Iarc', 'Vlrx', 'Vout']
# # scope_ch_arr = ['Varc', 'Vlrx', 'Iarc', 'Vrx_out']
# # scope_ch_arr = ['Varc', 'Iarc', 'Vout', 'Lrx']
# # scope_ch_arr = ['Istring', 'Lrx', 'Vou`t']

# # scope_ch_arr = ['Tx_injection']
# # scope_ch_arr = ['Inv_Lrx', 'Rx4_OUT', 'Varc', 'Iarc']
# scope_ch_arr = ['Lrx']

# spi_ch_arr = ['VRX']

# results_path=path+test_name
# # results_path=path+'/'+test_name+' - Python Results'
# # Open Results Lib
# if not os.path.exists(results_path):
#     os.makedirs(results_path)



# read_df_fft=True
# if read_df_fft:
#     EnergyDB=Read_df_fft(results_path+ '\\'+ file_name_arr[0] +' df_fft_mag.csv')
# plot=True  
# factor=1  
# zero_span_arr=[35e3]   
# SNR2,t_3snr=SNR(EnergyDB,zero_span_arr,results_path,plot,factor,'Reset reg')  
# # SNR,t_snr=SNR_Matced(EnergyDB,zero_span_arr,results_path,plot,factor,'Reset reg')  

# <codecell> Directory, Files, Tags, Data location
    
# for col in EnergyDB.columns[1:2]:
#     a=stage_1_energy_raise(EnergyDB[col],WindowSize=15,FilterSize=15,OverThresholdLimit=12)
#     print(a)
        
iac_arr_1=  [1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. ,
        1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. ,
        1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. ,
        1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. ,
        1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. ,
        1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. ,
        1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. ,
        1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 0.65, 0.75, 0.75, 0.75,
        0.75, 0.75, 0.75, 0.75, 0.75, 0.45, 0.75, 0.75, 0.75, 0.75, 0.75, 0.15, 0.75,
        0.75, 0.25, 0.75, 0.75,1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. ,
        1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. ,
        1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. ,
        
        1. , 1. , 1. , 1. , 1. , 1.,  1. , 1. , 1. , 1. , 1. , 1. , 1. ,
        1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. ,]
Iac_df = pd.DataFrame (iac_arr_1,columns=['Column_Name'])
# noam1=stage_1_Iac_raise(Iac_df,15,15,12)
# # noam=stage_1_Iac_jitter(Iac_df,15,15,4)
    
# <codecell> Directory, Files, Tags, Data location
