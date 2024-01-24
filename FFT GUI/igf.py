# -*- igf function library -*-
"""
    Created on Sun May 24 16:20:47 2020
    Function module which includes all relevant signal processing
    
    @author: Ilya Gluzman
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
# <codecell> Import Scope CSV file
def scope_CSV_to_df(path, file_name_arr, tag_arr, ch_arr):
    """
        Function import scope data CSV file into Pandas DataFrame
        
        Inputs:
        `path`              - Data files path (relatively to .py file location); String
        `file_name_arr`     - array of data files you want to analyse; String array 
        `tag_arr`           - Array of tags you want to attach to data files; String array 
        `ch_arr`            - Array of scope channels you want to analyse; String array 
        
        Returns:
            DF - Pandas Data Frame
        
        Example of usage : 
            df = igf.scope_CSV_to_df(path, file_name_arr, tag_arr, ch_arr)
    """
    print('--> Reading Scope Data CSV...')
    tss = time.time()
    filename=file_name_arr[0]
    df = pd.read_csv(path+filename+'.csv', header=0)
    df=df.add_prefix(tag_arr[0]+'_')
    df = df.rename(columns = {df.columns[0]: 'Time'})
    
    df['Time']=(df['Time']-df.loc[0,'Time'])
    
    for col in df.columns:
        if 'Unnamed' in col:
            # print(col)
            del df[col]
          
    dt=df.loc[1,'Time']-df.loc[0,'Time']
    Fs=int(1.0/dt)  # sampling rate
    Ts=1/Fs         # sampling interval
    df_len=len(df)
    df_time_len = max(df['Time'])-min(df['Time'])
    tmin=min(df['Time'])
    tmax=max(df['Time'])
    
    b=0
    for filename in file_name_arr[1:] :
        df_tmp = pd.read_csv(path+filename+'.csv', header=0)
            
        for col in df_tmp.columns:
            if 'Unnamed' in col:
                # print(col)
                del df_tmp[col]
            if 'Time' in col:
                # print(col)
                del df_tmp[col]   
        b=b+1
        
        df_tmp=df_tmp.add_prefix(tag_arr[b]+'_')
        df[df_tmp.columns] = df_tmp
    
    cols=['Time']
    for s in ch_arr:  
        c = df.columns[df.columns.str.contains(s)]
        cols = cols + c.tolist()
    df=df[cols]
      
    temp1='DF Tmin = '+str(tmin)+'[Sec]; '+'DF Tmax = '+str(tmax)+'[Sec]; \n'
    temp2='DF time length = '+str(round(df_time_len,5))+'[Sec] / ~'+str(round(df_time_len/60,4))+'[Min]; \n'
    text=temp1+temp2+'DF length = '+str(df_len/1000000)+'[Mega Samples];\n'+'DF Sampling rate = '+str(round((Fs/1000),0))+'[kSamp/sec]'+'; DF Sampling Interval = '+str(round((Ts*1000),3))+'[mSec]\n'
    
    print(text)
    print('Finished Reading Data')
    tee = time.time()
    print('--| Runtime = '+str(tee-tss)+' Sec')
    
    return (df)

# <codecell> Import SPI txt file
def spi_TXT_to_df(path, filename, tag_arr, spi_param_arr, spi_sample_rate):
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
    df = pd.read_csv(path+filename+'.txt', header=0)
    i=0
    for col in df.columns:
        if 'Unnamed' in col:
            # print(col)
            del df[col]
        else:
            df = df.rename(columns = {col: spi_param_arr[i]})
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

# <codecell> Chunk function Definition
def df_Chunk(df, t_start, t_end):
    """
    Function cuts data chunk and returns new DF in relevant time frame
    `df`        - Pandas Data Frame
    `t_start`   - Start time
    `t_end`     - End time
    
    Returns - DataFrame chunk
    
    Example of usage : 
        t_start = 1
        t_end = 5
        df1 = igf.df_Chunk(df, t_start, t_end, dt)
    """
    start = time.time()
    print('--> Cutting Data "Chunk"...')  
    dt=df['Time'][1001]-df['Time'][1002]
    df1=df[df['Time']>t_start-dt]
    df1=df1[df1['Time']<t_end]
    # df1=df1.astype(float)
    tmin=min(df1['Time'])
    tmax=max(df1['Time'])
    df1_time_len = round(tmax-tmin,2)
    df1_len=len(df1)
    
    temp='DF "chunk" time length = '+str(df1_time_len)+'[Sec]/~'+str(round(df1_time_len/60,2))+'[Min];\n'
    text=temp+'DF "chunk" Start time = '+str(tmin)+' [sec]'+'; DF "chunk" End time = '+str(tmax)+' [sec];\n'+'DF "chunk" length = '+str(df1_len/1000000)+' [Mega Samples]'
    print(text)
    end = time.time()
    print('--| Runtime = '+str(end-start)+' Sec')
    return (df1)

# <codecell> Time Reset 
def df_time_reset(dfx, x_col):
    """
        Function resets data chunk time vector
        
        `dfx`   - Pandas Data Frame
        `x_col` - exact Time col name  
        
        Example of usage : 
            df1 = igf.df_time_reset(df1,'Time')
    """
    dfx[x_col]=dfx[x_col]-dfx[x_col].iloc[0]  
    return (dfx)

# <codecell> Variance Calculation on DataFrame function
def df_var_calc(dfi, win_size, x_col):
    """
        Function calculates Rolling Variance of all data frame and return new DF
        
        `dfi`       - Pandas Data Frame
        `win_size`  - number of sample to calc Variance
        `x_col`     - exact x axis col name
        
        Example of usage : 
            win_size = 10000
            df_var = igf.df_var_calc(df1, win_size, 'Time')
    """
    start = time.time()
    print('\n--> Calculates Signal Variace...')  
    dfi_var=pd.DataFrame()
    dfi_var[x_col]=dfi[x_col]
    col=dfi.columns[1:]
    dfi_var[col]=dfi[col].rolling(window=win_size).var()
    dfi_var=dfi_var.add_suffix('_var')
    dfi_var=dfi_var.rename(columns={dfi_var.columns[0]: x_col})
    # dfi_var[col]=dfi_var.append(dfi_temp)
    print('Calculated variance for win size = '+str(win_size)+' Samples')
    
    end = time.time()
    print('--| Runtime = '+str(end-start)+' Sec')    
    
    return(dfi_var)

# <codecell> Average (Mean) Calculation on DataFrame function
def df_mean_calc(dfi, win_size, x_col):
    """
        Function calculates rolling average of all data frame and return new DF
        
        `dfi`       - Pandas Data Frame
        `win_size`  - number of sample to rolling average
        `x_col`     - exact x axis col name
        
        Example of usage : 
            df_mean = igf.df_mean_calc(df_fft, 100, 't')
    """
    start = time.time()
    print('\n--> Calculates Signal Sliding Average...')  
    dfi_mean=pd.DataFrame()
    dfi_mean[x_col]=dfi[x_col]
    col=dfi.columns[1:]
    dfi_mean[col]=dfi[col].rolling(window=win_size).mean()
    dfi_mean=dfi_mean.add_suffix('_mean')
    dfi_mean=dfi_mean.rename(columns={dfi_mean.columns[0]: x_col})
    # dfi_var[col]=dfi_var.append(dfi_temp)
    print('Calculated mean for win size = '+str(win_size)+' Samples')
    
    end = time.time()
    print('--| Runtime = '+str(end-start)+' Sec')
    
    return(dfi_mean)    

# <codecell> STD (Mean) Calculation on DataFrame function
def df_std_calc(dfi, win_size, x_col):
    """
        Function calculates rolling std of all data frame and return new DF
        
        `dfi`       - Pandas Data Frame
        `win_size`  - number of sample to rolling average
        `x_col`     - exact x axis col name
        
        Example of usage : 
            df_std = igf.df_std_calc(df, 100, 't')
    """
    start = time.time()
    print('\n--> Calculates Signal Sliding std...')  
    dfi_std=pd.DataFrame()
    dfi_std[x_col]=dfi[x_col]
    col=dfi.columns[1:]
    dfi_std[col]=dfi[col].rolling(window=win_size).std()
    dfi_std=dfi_std.add_suffix('_std')
    dfi_std=dfi_std.rename(columns={dfi_std.columns[0]: x_col})
    print('Calculated std for win size = '+str(win_size)+' Samples')
    
    end = time.time()
    print('--| Runtime = '+str(end-start)+' Sec')
    
    return(dfi_std)    
# <codecell> Data dB calculation
def df_dB_calc(dfi, x_col):
    """
        Function calculates 20*Log10(dfi) dataframe columns
        `dfi`       - input Pandas Data Frame
        `x_col`     - exact x axis col name

    """
    start = time.time()
    print('\n--> Calculates 20*Log10('+dfi.columns[1]+')...')  
    dfi_dB=pd.DataFrame()
    dfi_dB[x_col]=dfi[x_col]
    col=dfi.columns[1:]
    dfi[dfi==0]=10**-12
    dfi_dB[col]=20*np.log10(dfi[col])
    dfi_dB=dfi_dB.add_suffix('_[dB]')
    dfi_dB=dfi_dB.rename(columns={dfi_dB.columns[0]: x_col})
    end = time.time()
    print('--| Runtime = '+str(end-start)+' Sec')
    
    return(dfi_dB)    
           
# <codecell> STFT Transform Calculation function
def df_stft_RAW_calc(dfx, Fs, fft_win, fft_win_overlap, ch_arr):
    """
        Function calculates spectromgram with SciPy "stft" function
        Inputs:
            `dfx`               - Pandas Data Frame
            `Fs`                - Data sampling frequency
            `fft_win'           - fft window size
            `fft_win_overlap`   - exact x axis col name
            `ch_arr`            - list of mesured scope channels 
        Outputs:
            `t`                 - time vector
            `f`                 - frequency vector
            `Zdata`             - Tuplet of RAW magnitudes maps in dBm
            `Name_arr`          - Array of Z names in relevant order
        Example of usage : 
            (df_fft, df_MH, df_AVG, t, f, Z_arr, Name_arr) = igf.df_spectrogram_calc(dfx, fft_win, fft_win_overlap, zero_span_arr, ch_arr)
    """    
    start = time.time()
    print('\n--> Calculates Short Time Furier Transform (STFT)...')  
    Z_arr=[]
    Name_arr=[]
    print_flag=0
    for meas_sig in ch_arr:
        N = fft_win #Number of point in the fft
        w = signal.hamming(N) #FFT window
        # w = signal.blackman(N) #FFT window
        for col_fft in dfx.columns[1:]:
            # f_arr=zero_span_arr
            if meas_sig in col_fft:
                sig = dfx[col_fft]   
                f, t, Zxx = signal.stft(sig, Fs, window=w,nperseg=N, noverlap=fft_win_overlap)
                gc.collect()           
                Z_arr.append(Zxx)
                Name_arr.append(col_fft)

                if print_flag==0:
                    txt = meas_sig+' FFT Window Time Frame = '+str(round((N/Fs*1000),3))+' [mSec]'
                    print(txt)
                    print(meas_sig+' Number of Time indexes = '+str(len(t))+'   Time step = '+str(round(((t[1]-t[0])*1000),3))+' [mSec]')
                    print(meas_sig+' Number of Freq indexes = '+str(len(f))+'   Freq step = '+str(round((f[1]-f[0]),3))+' [Hz]')
                    print_flag=1
        
    end = time.time()
    print('--| Runtime = '+str(end-start)+' Sec')
    
    return(t, f, Z_arr, Name_arr)  

# <codecell> Z Magnitude Calculation function
def Z_mag_calc(Zraw_arr):
    """
        Function calculates magnitudes of STFT output
        Inputs:
            `Zraw_arr`          - Tuplet of comples STFT results matrices 
        Outputs:
            `ZdBm_arr`          - Tuplet of STFT magnitudes in dBm
        Example of usage : 
            ZdBm_arr = igf.Z_mag_calc(Zraw_arr)
    """    
    start = time.time()
    print('\n--> Calculates magnitude in dBm of STFT Zraw matrix.') 

    ZdBm_arr = []
    for i in range(len(Zraw_arr)):
        Zxx=Zraw_arr[i]
        Zxx=2*np.abs(Zxx)
        Zxx[Zxx==0]=10**-12
        Zxx = (20*np.log10(Zxx/10))+30
        ZdBm_arr.append(Zxx)
        del Zxx
        gc.collect()                
    end = time.time()
    print('--| Runtime = '+str(end-start)+' Sec')
    
    return(ZdBm_arr)    

# <codecell> Z Phase Calculation function
def Z_phase_calc(Zraw_arr, phase_unwrap):
    """
        Function calculates spectromgram with SciPy "stft" function
        Inputs:
            `Zraw_arr`          - Tuplet of comples STFT results matrices 
        Outputs:
            `Zphase_arr`          - Tuplet of STFT magnitudes in dBm
        Example of usage : 
            Zphase_arr = igf.Z_phase_calc(Zraw_arr)
    """    
    start = time.time()
    if phase_unwrap:
        print('\n--> Calculates unwraped phase in degrees of STFT Zraw matrix.')
    else:
        print('\n--> Calculates phase in degrees of STFT Zraw matrix.')
    Zphase_arr = []
    for i in range(len(Zraw_arr)):
        Zxx=Zraw_arr[i]
        Zxx=np.angle(Zxx, deg=True)
        # Zphase_arr.append(Zxx)
        if phase_unwrap:
            Zxx=np.unwrap(Zxx)
        Zphase_arr.append(Zxx)
        del Zxx
        gc.collect()                
    end = time.time()
    print('--| Runtime = '+str(end-start)+' Sec')
    
    return(Zphase_arr)    

# <codecell> STFT Transform and Magnitude in dBm Calculation function
def df_stft_mag_calc(dfx, Fs, fft_win, fft_win_overlap, ch_arr):
    """
        Function calculates spectromgram with SciPy "stft" function
        Inputs:
            `dfx`               - Pandas Data Frame
            `Fs`                - Data sampling frequency
            `fft_win'           - fft window size
            `fft_win_overlap`   - exact x axis col name
            `ch_arr`            - list of mesured scope channels 
        Outputs:
            `t`                 - time vector
            `f`                 - frequency vector
            `Zdata`             - Tuplet of RAW magnitudes maps in dBm
            `Name_arr`          - Array of Z names in relevant order
        Example of usage : 
            (df_fft, df_MH, df_AVG, t, f, Z_arr, Name_arr) = igf.df_spectrogram_calc(dfx, fft_win, fft_win_overlap, zero_span_arr, ch_arr)
    """    
    start = time.time()
    print('\n--> Calculates Short Time Furier Transform (STFT)...')  
    Z_arr=[]
    Name_arr=[]
    print_flag=0
    for meas_sig in ch_arr:
        N = fft_win #Number of point in the fft
        w = signal.hamming(N) #FFT window
        # w = signal.blackman(N) #FFT window
        for col_fft in dfx.columns[1:]:
            # f_arr=zero_span_arr
            if meas_sig in col_fft:
                sig = dfx[col_fft]   
                f, t, Zxx = signal.stft(sig, Fs, window=w,nperseg=N, noverlap=fft_win_overlap)
                # Zxx=Zxx.astype('complex64')
                Zxx=2*np.abs(Zxx)
                Zxx[Zxx==0]=10**-12
                # Zxx=Zxx.astype('float32')
                Z_dBm = (20*np.log10(Zxx/10))+30
                # Z_dBm = Z_dBm.astype('float32')
                del Zxx
                gc.collect()           
                Z_arr.append(Z_dBm)
                Name_arr.append(col_fft)

                if print_flag==0:
                    txt = meas_sig+' FFT Window Time Frame = '+str(round((N/Fs*1000),3))+' [mSec]'
                    print(txt)
                    print(meas_sig+' Number of Time indexes = '+str(len(t))+'   Time step = '+str(round(((t[1]-t[0])*1000),3))+' [mSec]')
                    print(meas_sig+' Number of Freq indexes = '+str(len(f))+'   Freq step = '+str(round((f[1]-f[0]),3))+' [Hz]')
                    print_flag=1
        
    end = time.time()
    print('--| Runtime = '+str(end-start)+' Sec')
    
    return(t, f, Z_arr, Name_arr)          

# <codecell> ZeroSpan Calculation function
def ZeroSpan_calc(Z_arr, Z_name_arr, t, f, zero_span_arr, ch_arr):
    """
        Function calculates spectromgram with SciPy "stft" function
        Inputs:
            `Z_arr`             - Tuplet of RAW magnitudes maps in dBm
            `Z_name_arr`        - Array of Z names in relevant order
            `t`                 - time vector
            `f`                 - frequency vector
            `zero_span_arr`     - list of frequencoes to perform Zero Span Calc
            `ch_arr`            - list of mesured scope channels 
        Outputs:
            `df_fft`            - Zero Span results Pandas Data Frame
        Example of usage : 
            (df_fft, df_MH, df_AVG, t, f, Z_arr, Name_arr) = igf.df_spectrogram_calc(dfx, fft_win, fft_win_overlap, zero_span_arr, ch_arr)
    """    
    start = time.time()
    print('\n--> Calculates ZeroSpan time domain energy levels for given STFT matrix (Z array) and frequencies...')  
    df_fft = pd.DataFrame(columns=['t'])
    for meas_sig in ch_arr:
        flag=0
        f_arr=zero_span_arr
        z_ind=0
        for col_fft in Z_name_arr:
            Z_dBm = Z_arr[z_ind]
            z_ind=z_ind+1
            if meas_sig in col_fft:
                for ff in f_arr:
                    ind = int(ff/(f[1]-f[0]))
                    print(str(ind))
                    z_dBm = (Z_dBm[ind,:])
                    df_fft[col_fft+' @ '+str(round(float(f[ind]/1000),1))+' [kHz]']=z_dBm
                    # print(str(ind))
           
                if flag==0:
                    flag=1
                    df_fft['t']=t
        
    end = time.time()
    print('--| Runtime = '+str(end-start)+' Sec')
    
    return(df_fft)               

# <codecell> ZeroSpan Covariance and Correlation function
def ZeroSpan_xcor_calc(df_fft, win_size, xcor_freq_pair):
    """
        Function calculates covariance and correlation on 2 ZeroSpan results
        Inputs:
            `df_fft`            - Zero Span resutls data frame
            `win_size`          - number of samples of covariance/correlation calcs
            `xcor_freq_pair`    - pair of freqs for xcorr calcs
        Outputs:
            `df_xcor`           - dataframe with time, source, corralation and covariance vectors
    """ 
    fp=xcor_freq_pair
    df_xcor = pd.DataFrame(columns=['t'])
    df_xcor['t'] = df_fft['t']
    
    for col in df_fft.columns:
        if fp[0] in col or fp[1] in col:
            if 'mean' not in col:
                df_xcor[col]=df_fft[col]
                
    
    cols=df_xcor.columns[1:3]
    a=cols[0]
    b=cols[1]
    c=cols[0]+' vs ' + fp[1] + ' [kHz]'+ ' - corr'
    d=cols[0]+' vs ' + fp[1] + ' [kHz]'+ ' - cov'
    df_xcor[c]=df_xcor[cols].rolling(window=win_size).corr().unstack()[a][b]
    df_xcor[d]=df_xcor[cols].rolling(window=win_size).cov().unstack()[a][b]       
    
    return(df_xcor)
    
# <codecell> Goertzel (ZeroSpan DFT) function 
def goertzel(samples, sample_rate, *freqs):
    """
    Implementation of the Goertzel algorithm, useful for calculating individual
    terms of a discrete Fourier transform.

    `samples` is a windowed one-dimensional signal originally sampled at `sample_rate`.

    The function returns 2 arrays, one containing the actual frequencies calculated,
    the second the coefficients `(real part, imag part, power)` for each of those frequencies.
    For simple spectral analysis, the power is usually enough.

    Example of usage : 

        # calculating frequencies in ranges [400, 500] and [1000, 1100]
        # of a windowed signal sampled at 44100 Hz

        freqs, results = goertzel(some_samples, 44100, (400, 500), (1000, 1100))
    """
    window_size = len(samples)
    f_step = sample_rate / float(window_size)
    f_step_normalized = 1.0 / window_size

    # Calculate all the DFT bins we have to compute to include frequencies
    # in `freqs`.
    bins = set()
    for f_range in freqs:
        f_start, f_end = f_range
        k_start = int(math.floor(f_start / f_step))
        k_end = int(math.ceil(f_end / f_step))

        if k_end > window_size - 1: raise ValueError('frequency out of range %s' % k_end)
        bins = bins.union(range(k_start, k_end))

    # For all the bins, calculate the DFT term
    n_range = range(0, window_size)
    freqs = []
    results = []
    for k in bins:

        # Bin frequency and coefficients for the computation
        f = k * f_step_normalized
        w_real = 2.0 * math.cos(2.0 * math.pi * f)
        w_imag = math.sin(2.0 * math.pi * f)

        # Doing the calculation on the whole sample
        d1, d2 = 0.0, 0.0
        for n in n_range:
            y  = samples[n] + w_real * d1 - d2
            d2, d1 = d1, y

        # Storing results `(real part, imag part, power)`
        results.append((
            0.5 * w_real * d1 - d2, w_imag * d1,
            d2**2 + d1**2 - w_real * d1 * d2)
        )
        freqs.append(f * sample_rate)
    return freqs, results

# <codecell> Spliding Spectrum Calculation Function
def sliding_spectrum(Z, t, f, win_size, win_overlap, meas_sig):
    """
    Function performing sliding MaxHold and AVG functios on Spectrogram and returns data frame of spectrum frames
   
    `Z`             - Results matrix
    `t`             - time vector      (series)
    `f`             - frequency vector (series)
    `win_size`      - number of samples to calculate Max hold and AVG (integer)
    `win_overlap`   - number of samples to overlap windows (integer)    
    `meas_sig`      - presented signal name (string)
    
    Example of usage:
        (df_MH, df_AVG, t_list)=igf.sliding_spectrum(Z, t, f, win_size, win_overlap, meas_sig)
    """

    print('\n--> Calculating '+meas_sig+' "Sliding" Spectrum...')
    start = time.time()
    
    if win_size > len(t)-1:
        print("win size is to long! Please reduce the number of samples to be below "+str(len(t))+" value")
        return()
    
    if win_size < win_overlap:
        print("win overlap size is to long! Please reduce the number of samples to be below "+str(win_size)+" value")
        return()
    
    df_MH = pd.DataFrame()
    df_AVG = pd.DataFrame()
    
    # df_MH = pd.DataFrame(columns=['f'])
    # df_AVG = pd.DataFrame(columns=['f'])   
    # df_MH['f']=f
    # df_AVG['f']=f
    
    win_size=int(win_size)
    win_overlap=int(win_overlap)
    
    # tstep=t[1]-t[0]
    i1=0
    i2=win_size
    di=win_size-win_overlap
    t_list = t[i2::di]
    
    while i2 < len(t):
    
        Zt = Z[:,i1:i2]
        Zt_AVG=Zt.mean(axis=1)    
        Zt_MH=Zt.max(axis=1) 
        # Zt_MH=Zt.std(axis=1)
        
        t1=t[i1]
        t2=t[i2]
        col_name=meas_sig+' @ t = '+str(round(t1,3))+'-'+str(round(t2,3))+' [S]'
        df_MH[col_name]=Zt_MH
        df_AVG[col_name]=Zt_AVG
        
        i1=i1+di
        i2=i2+di


    end = time.time()
    print('--| Runtime = '+str(end-start)+' Sec')
    
    return(df_MH, df_AVG,t_list)        

# <codecell> Sliding Goertzel (ZS DFT) calculation
def sliding_goertzel(dfx, t_col_name, sig_col_name, FFT_freq, win_size, win_overlap):
    """
    Function performing sliding Goertzel functios on RAW data frame and returns Goertzel for specific frequency 
   
    `dfx`           - input Data Frame --> Column from DF
    `t_col_name`    - name of time vector in data frame  (str)
    `sig_col_name`  - presented signal name/scope channel (string)
    `FFT_freq'      - value of required FFT freq.
    `win_size`      - number of samples to calculate FFT window (integer)
    `win_overlap`   - number of samples to overlap windows (integer)    
    
    (results_df) = igf.sliding_goertzel(dfx, t_col_name, sig_col_name, FFT_freq, win_size, win_overlap)
    """

    print('\n--> Calculating '+sig_col_name+' "Sliding" Goertzel...')
    start = time.time()
    
    # get signal vector and sample rate
    sx=dfx[sig_col_name].to_numpy()
    
    # get time vector and sample rate
    t=dfx[t_col_name].to_numpy()
    sample_rate = 1/(t[2]-t[1])
    
    if win_size > len(t)-1:
        print("win size is to long! Please reduce the number of samples to be below "+str(len(t))+" value")
        return()
    
    if win_size < win_overlap:
        print("win overlap size is to long! Please reduce the number of samples to be below "+str(win_size)+" value")
        return()
    
    # calculate frequency resolution
    f_resolution = sample_rate/win_size
    f1=FFT_freq+(f_resolution/2)
    f2=FFT_freq+(f_resolution)
    results_df = pd.DataFrame(columns=['t', 'data'])
    
    win_size=int(win_size)
    win_overlap=int(win_overlap)   

    i1=0
    i2=win_size
    di=win_size-win_overlap
    t_list = t[i2::di]
    results_df['t']=t_list
    df_ind=0
    while i2 < len(t):
        sxt = sx[i1:i2]
        freqs, results = goertzel(sxt, sample_rate, (f1, f2))
        res_P = results[0][2]
        results_df.loc[df_ind,'data']=res_P
        df_ind=df_ind+1
        i1=i1+di
        i2=i2+di
        
        
    col_name=sig_col_name+' @ f = '+str(round((freqs[0]/1000),2))+' [kHz]'
    results_df=results_df.rename(columns={'t': 'Time', 'data': col_name})
    
    end = time.time()
    print('--| Runtime = '+str(end-start)+' Sec')
    
    return(results_df) 
             
# <codecell> Plot Data function
def data_plot(dfx, data_name, x_col ,max_plot_res, meas_sig, auto_open_plot, file_on, results_path):
    """
        Function genearates Scattergl graph data array and if chose generate html plot
        `dfx`           - input data frame with requested data to plot
        `data name`     - text string to define file and graph name
        `x_col`         - plot X axis column
        `max_plot_res`  - max plot resolution
        `meas_sig`      - requested channel or data column to plot
        `auto_open_plot`- True/False for Auto Open of plots in web browser
        `file_on`       - True/False create html file   
        `results_path`  - results path to store html files
    
        Example of usage :
            (data_out) = igf.data_plot(dfx, data_name, x_col ,max_plot_res, meas_sig, auto_open_plot, file_on, results_path)
    """
    
    start = time.time()
    print('\n--> Generating "'+data_name+'" plot...')  
    plot_res=int(len(dfx)/max_plot_res)
    if plot_res==0:
        plot_res=1
    data_out=[x_col]
    # fig_out=[]
    
    data=[]
    for col in dfx:
        if meas_sig in col:
            data.append( 
                go.Scattergl(
                x=dfx[x_col].iloc[::plot_res],
                y=dfx[col].iloc[::plot_res],
                name= col+' - '+data_name,
                mode = 'lines',
                #mode = 'markers',
                hoverlabel = dict( namelength = -1) 
                )
            )

    fig = go.FigureWidget(data)
    config = {'scrollZoom': True}
    fig['layout'].update(title=meas_sig+' '+data_name)
    fig['layout'].update(xaxis=dict(title = x_col))
    data_out.append(data)
    # fig_out.append(fig)
    txt=results_path+'/'+meas_sig+' '+data_name
    if file_on:
        print("Generated plot file")
        py.offline.plot(fig,auto_open = auto_open_plot, config=config, filename=txt+'.html')    
    
    end = time.time()
    print('--| Runtime = '+str(end-start)+' Sec')
    
    return(data_out)

# <codecell> Plot N-panes Data function Definition 
def data_pane_plot(data_name, data_list, name_list, plot_on, x_sync, tag_arr, channel, results_path):
    """
    Function plot "N" data panes 
    N between 1-4
    `data name` - tesxt string to define file and graph name
    `data list` - list of plot data which generated by data_plot()
    `name_list` - list of test strings to define pane names
    `plot_on`   - True/False for Auto Open of plots in web browser
    `x_sync`    - True/False shared_xaxes configuration   
    `tag_arr`   - array of filenames
    `channel`    - array of scope channels to be adressed
    `results_path`  - results path to store html files

    Example of usage : 
        data_list=[scope_plot_data, var_plot_data, AVG_plot_data]
        name_list=['Scope', 'Var', 'Mean Var']
        igf.data_pane_plot(data_name, data_list, name_list, plot_on, x_sync, tag_arr, channel, results_path)
    """
    
    start = time.time()
    print('\n--> Generating "'+data_name+'" plot with panes...') 
    n_case = len(data_list)
    
    xaxes_sync=x_sync
    yaxes_sync=False
    
    spacing=0.1
    for k in range(1):
        if n_case != len(name_list):
            print("Input strings' length mismatch!!! Please verify inputs")
            return()
            
        if n_case==1:
            # print('case 1')
            fig = py.subplots.make_subplots(rows=n_case, cols=1, subplot_titles=(name_list[0]),
                                specs=[[{"secondary_y": True}]],
                                  shared_xaxes=xaxes_sync, shared_yaxes=yaxes_sync,
                                  vertical_spacing=spacing)
        elif n_case==2:
            # print('case 2')
            fig = py.subplots.make_subplots(rows=n_case, cols=1, subplot_titles=(name_list[0], name_list[1]),
                                specs=[[{"secondary_y": True}], [{"secondary_y": True}]],
                                  shared_xaxes=xaxes_sync, shared_yaxes=yaxes_sync,
                                  vertical_spacing=spacing)
        elif n_case==3:
            # print('case 3')
            fig = py.subplots.make_subplots(rows=n_case, cols=1, subplot_titles=(name_list[0], name_list[1], name_list[2]),
                                specs=[[{"secondary_y": True}], [{"secondary_y": True}], [{"secondary_y": True}]],
                                  shared_xaxes=xaxes_sync, shared_yaxes=yaxes_sync,
                                  vertical_spacing=spacing)
        elif n_case==4:
            # print('case 4')
            fig = py.subplots.make_subplots(rows=n_case, cols=1, subplot_titles=(name_list[0], name_list[1], name_list[2], name_list[3]),
                                specs=[[{"secondary_y": True}], [{"secondary_y": True}], [{"secondary_y": True}], [{"secondary_y": True}]],
                                  shared_xaxes=xaxes_sync, shared_yaxes=yaxes_sync,
                                  vertical_spacing=spacing)
        else:
            print('Too match traces!!!')
            return()
        
        n=1
        for trace in data_list:
            loop_count = len(trace[k+1])
            for i in range(loop_count):
                fig.add_trace(trace[k+1][i], n, 1, secondary_y=False)
                fig.update_xaxes(title_text=trace[0], row=n, col=1)
            n=n+1
    
        config = {'scrollZoom': True}
        fig['layout'].update(title=channel+' '+data_name)
        
        txt=results_path+'/'+' '+channel+' '+data_name
        py.offline.plot(fig,auto_open = plot_on, config=config, filename=txt+'.html')
        
    end = time.time()
    print('--| Runtime = '+str(end-start)+' Sec')        

# <codecell> Plot Spectrogram "Heatmap"
def scpectrogram_plot(Z, t, f, max_plot_res, fmax, fmin, t_start, plot_on, results_path, meas_sig):
    """
        Function generates spectrogram plot
       
        `Z`             - Results matrix
        `t`             - time vector      (series)
        `f`             - frequency vector (series)
        `max_plot_res`  - max plot resolution (float)
        `fmax`          - max freq (float) 
        `fmin`          - min freq (float)
        `t_start`       - chunk start time (float)
        `plot_on`       - True/False for Auto Open of plots in web browser (boolean)
        `results_path`  - Test resutls path (string)
        `meas_sig`      - presented signal name (string)
    """
    print('\n--> Generating "'+meas_sig+'" Spectrogram...')
    start = time.time()    
    f_min_ind=int(fmin/(f[1]-f[0]))
    f_max_ind=int(fmax/(f[1]-f[0]))+1
    
    spec_res=int(len(t)/max_plot_res)
    freq_res=int(len(f)/max_plot_res)

    if spec_res==0:
        spec_res=1
    if freq_res==0:
        freq_res=1

    trace = [go.Heatmap(
        x= t[::spec_res]+t_start,
        y= f[f_min_ind:f_max_ind],
        z= Z[f_min_ind:f_max_ind,::spec_res],
        colorscale='Jet',
        )]
    
    layout = go.Layout(
        title = meas_sig+' Spectrogram [dBm]',
        yaxis = dict(title = 'Frequency [Hz]'), # x-axis label
        xaxis = dict(title = 'Time [sec]'), # y-axis label
        )
    
    fig = go.Figure(data=trace, layout=layout)
    config = {'scrollZoom': True}
    
    txt=results_path+'/'+meas_sig
    py.offline.plot(fig, auto_open = plot_on, config=config, filename=txt+' Spectrogram.html')

    end = time.time()
    print('--| Runtime = '+str(end-start)+' Sec')
