import os
import threading
import os
import time
import pandas as pd
import gc
# import pywt
import numpy as np
import plotly as py
import plotly.graph_objs as go
import math
import statistics
from scipy import signal
import matplotlib.pyplot as plt
from my_pyplot import plot as _P, print_chrome as _PC, clear as _PP, print_lines as _PL, send_mail as _SM
import Plot_Graphs_with_Sliders as _G
import my_tools

return_min_with_max_avg = False
voltage_peak_to_peak = 2.5
adc_bits = 14
# que = queue.Queue()


# <codecell> Import Scope CSV file
def scope_CSV_to_df(path, filename, tag_arr, ch_arr, has_time_col, fs, extension='.txt', delimiter='\t', skiprows=0):
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
    tss = time.time()
    df = pd.read_csv(path + filename + extension, header=0, delimiter=delimiter, skiprows=skiprows)
    #   df=df.add_prefix(tag_arr[0]+'_')
    #   df = df.rename(columns = {df.columns[0]: 'Time'})
    if has_time_col:
        df['Time'] = (df['Time'] - df.loc[0, 'Time'])
    else:
        df['Time'] = df.index * (1 / fs)
    for col in df.columns:
        if 'Unnamed' in col:
            del df[col]

    dt = df.loc[1, 'Time'] - df.loc[0, 'Time']
    Fs = int(1.0 / dt)  # sampling rate
    Ts = 1 / Fs  # sampling interval
    df_len = len(df)
    df_time_len = max(df['Time']) - min(df['Time'])
    tmin = min(df['Time'])
    tmax = max(df['Time'])

    b = 0
    # for filename in file_name_arr[1:] :
    #     df_tmp = pd.read_csv(path+filename+'.csv', header=0)
    #
    #     for col in df_tmp.columns:
    #         if 'Unnamed' in col:
    #             # print(col)
    #             del df_tmp[col]
    #         if 'Time' in col:
    #             # print(col)
    #             del df_tmp[col]
    #     b=b+1
    #
    #     df_tmp=df_tmp.add_prefix(tag_arr[b]+'_')
    #     df[df_tmp.columns] = df_tmp
    #
    # cols=['Time']
    # for s in ch_arr:
    #     c = df.columns[df.columns.str.contains(s)]
    #     cols = cols + c.tolist()
    # df=df[cols]

    temp1 = 'DF Tmin = ' + str(tmin) + '[Sec]; ' + 'DF Tmax = ' + str(tmax) + '[Sec]; \n'
    temp2 = 'DF time length = ' + str(round(df_time_len, 5)) + '[Sec] / ~' + str(
        round(df_time_len / 60, 4)) + '[Min]; \n'
    text = temp1 + temp2 + 'DF length = ' + str(df_len / 1000000) + '[Mega Samples];\n' + 'DF Sampling rate = ' + str(
        round((Fs / 1000), 0)) + '[kSamp/sec]' + '; DF Sampling Interval = ' + str(round((Ts * 1000), 3)) + '[mSec]\n'

    tee = time.time()

    return (df)


# <codecell> Import SPI txt file
def spi_TXT_to_df(path, filename, tag_arr, spi_param_arr, spi_sample_rate, plot_raw=True):
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
    tsss = time.time()
    Ts = 1 / spi_sample_rate
    Fs = spi_sample_rate
    df_spi = pd.DataFrame(columns=['Time'])
    df = pd.read_csv(path + filename + '.txt', header=0)
    i = 0

    for col in df.columns:
        if 'Unnamed' in col:
            del df[col]
        # else:
        #     df = df.rename(columns={col: spi_param_arr[i]})
        #     i = i + 1
    #
    df_spi['Time'] = (df.index) * Ts
    df = df.loc[:, ~df.columns.duplicated()]
    if plot_raw == False:
        for col in df.columns:
            if ('Vin' in col or 'Vout' in col):
                V_quantization = 1 / (2 ** 6)
                df_spi[col] = df[col] * V_quantization
            elif ('Iin' in col or 'IL' in col):
                V_quantization = 1 / (2 ** 9)
                df_spi[col] = df[col] * V_quantization
            else:
                V_quantization = 1 / (2 ** 12)
                V_quantization = 1
                df_spi[col] = df[col] * V_quantization
    else:
        for col in df.columns:
            df_spi[col] = df[col]
    df_len = len(df_spi)
    df_time_len = max(df_spi['Time']) - min(df_spi['Time'])
    tmin = min(df_spi['Time'])
    tmax = max(df_spi['Time'])

    temp1 = 'DF Tmin = ' + str(tmin) + '[Sec]; ' + 'DF Tmax = ' + str(tmax) + '[Sec]; \n'
    temp2 = 'DF time length = ' + str(round(df_time_len, 5)) + '[Sec] / ~' + str(
        round(df_time_len / 60, 4)) + '[Min]; \n'
    text = temp1 + temp2 + 'DF length = ' + str(df_len / 1000000) + '[Mega Samples];\n' + 'DF Sampling rate = ' + str(
        round((Fs / 1000), 0)) + '[kSamp/sec]' + '; DF Sampling Interval = ' + str(round((Ts * 1000), 3)) + '[mSec]'

    teee = time.time()

    return (df_spi)


def kill_chrome(bool):
    if bool:
        os.system("taskkill /im chrome.exe /f")


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
    dt = df['Time'][1001] - df['Time'][1002]
    df1 = df[df['Time'] > t_start - dt]
    df1 = df1[df1['Time'] < t_end]
    # df1=df1.astype(float)
    tmin = min(df1['Time'])
    tmax = max(df1['Time'])
    df1_time_len = round(tmax - tmin, 2)
    df1_len = len(df1)

    temp = 'DF "chunk" time length = ' + str(df1_time_len) + '[Sec]/~' + str(round(df1_time_len / 60, 2)) + '[Min];\n'
    text = temp + 'DF "chunk" Start time = ' + str(tmin) + ' [sec]' + '; DF "chunk" End time = ' + str(
        tmax) + ' [sec];\n' + 'DF "chunk" length = ' + str(df1_len / 1000000) + ' [Mega Samples]'
    end = time.time()
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
    dfx[x_col] = dfx[x_col] - dfx[x_col].iloc[0]
    return (dfx)


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
    dfi_var = pd.DataFrame()
    dfi_var[x_col] = dfi[x_col]
    col = dfi.columns[1:]
    dfi_var[col] = dfi[col].rolling(window=win_size).var()
    dfi_var = dfi_var.add_suffix('_var')
    dfi_var = dfi_var.rename(columns={dfi_var.columns[0]: x_col})
    # dfi_var[col]=dfi_var.append(dfi_temp)

    end = time.time()

    return (dfi_var)


def df_dB_calc(dfi, x_col):
    """
        Function calculates 20*Log10(dfi) dataframe columns
        `dfi`       - input Pandas Data Frame
        `x_col`     - exact x axis col name

    """
    start = time.time()
    dfi_dB = pd.DataFrame()
    dfi_dB[x_col] = dfi[x_col]
    col = dfi.columns[1:]
    dfi[dfi == 0] = 10 ** -12
    dfi_dB[col] = 20 * np.log10(dfi[col])
    dfi_dB = dfi_dB.add_suffix('_[dB]')
    dfi_dB = dfi_dB.rename(columns={dfi_dB.columns[0]: x_col})
    end = time.time()

    return (dfi_dB)


# <codecell> STFT Transform Calculation function

# <codecell> Plot Data function
def data_plot(dfx, data_name, x_col, max_plot_res, meas_sig, auto_open_plot, file_on, results_path):
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
    plot_res = int(len(dfx) / max_plot_res)
    if plot_res == 0:
        plot_res = 1
    plot_res = 1
    data_out = [x_col]
    # fig_out=[]
    data = []
    for col in dfx:
        if meas_sig in col:
            data.append(
                go.Scattergl(
                    x=dfx[x_col].iloc[::plot_res],
                    y=dfx[col].iloc[::plot_res],
                    name=col + ' - ' + data_name,
                    mode='lines',
                    # mode = 'markers',
                    hoverlabel=dict(namelength=-1)
                )
            )

    fig = go.FigureWidget(data)
    config = {'scrollZoom': True, 'editable': True}
    fig['layout'].update(title=meas_sig + ' ' + data_name)
    fig['layout'].update(xaxis=dict(title=x_col))
    data_out.append(data)
    # fig_out.append(fig)
    txt = results_path + '/' + meas_sig + ' ' + data_name
    if file_on:
        # fig.write_image(txt + '.jpeg')
        py.offline.plot(fig, auto_open=auto_open_plot, config=config, filename=txt + '.html')

    end = time.time()

    return (data_out)


# <codecell> Plot N-panes Data function Definition
def data_pane_plot(data_name, data_list, name_list, plot_on, x_sync, tag_arr, channel, results_path, ver=1):
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
    n_case = len(data_list)

    xaxes_sync = x_sync
    yaxes_sync = False

    spacing = 0.1
    for k in range(1):
        if n_case != len(name_list):
            return ()

        if n_case == 1:
            fig = py.subplots.make_subplots(rows=n_case, cols=1, subplot_titles=(name_list[0]),
                                            specs=[[{"secondary_y": True}]],
                                            shared_xaxes=xaxes_sync, shared_yaxes=yaxes_sync,
                                            vertical_spacing=spacing)
        elif n_case == 2:
            fig = py.subplots.make_subplots(rows=n_case, cols=1, subplot_titles=(name_list[0], name_list[1]),
                                            specs=[[{"secondary_y": True}], [{"secondary_y": True}]],
                                            shared_xaxes=xaxes_sync, shared_yaxes=yaxes_sync,
                                            vertical_spacing=spacing)
        elif n_case == 3:
            fig = py.subplots.make_subplots(rows=n_case, cols=1,
                                            subplot_titles=(name_list[0], name_list[1], name_list[2]),
                                            specs=[[{"secondary_y": True}], [{"secondary_y": True}],
                                                   [{"secondary_y": True}]],
                                            shared_xaxes=xaxes_sync, shared_yaxes=yaxes_sync,
                                            vertical_spacing=spacing)
        elif n_case == 4:
            fig = py.subplots.make_subplots(rows=n_case, cols=1,
                                            subplot_titles=(name_list[0], name_list[1], name_list[2], name_list[3]),
                                            specs=[[{"secondary_y": True}], [{"secondary_y": True}],
                                                   [{"secondary_y": True}], [{"secondary_y": True}]],
                                            shared_xaxes=xaxes_sync, shared_yaxes=yaxes_sync,
                                            vertical_spacing=spacing)
        else:
            return ()

        n = 1
        for trace in data_list:
            loop_count = len(trace[k + 1])
            for i in range(loop_count):
                fig.add_trace(trace[k + 1][i], n, 1, secondary_y=False)
                fig.update_xaxes(title_text=trace[0], row=n, col=1)
            n = n + 1

        config = {'scrollZoom': True, 'editable': True}
        import os
        if ver == 1:
            fig['layout'].update(title=data_name + ': ' + results_path.split('\\')[-1])
            txt = results_path + '\\' + data_name
        else:
            fig['layout'].update(title=channel + ' ' + data_name)
            txt = results_path + '\\' + data_name + ' ' + channel
        this_dir = os.path.dirname(os.path.abspath(__file__))

        py.offline.plot(fig, auto_open=plot_on, config=config, filename=txt + '.html')

    end = time.time()


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
    df_fft = pd.DataFrame(columns=['t'])
    for meas_sig in ch_arr:
        flag = 0
        f_arr = zero_span_arr
        z_ind = 0
        for col_fft in Z_name_arr:
            Z_dBm = Z_arr[z_ind]
            z_ind = z_ind + 1
            if meas_sig in col_fft:
                for ff in f_arr:
                    ind = int(ff / (f[1] - f[0]))
                    z_dBm = (Z_dBm[ind, :])
                    df_fft[col_fft + ' @ ' + str(round(float(f[ind] / 1000), 1)) + ' [kHz]'] = z_dBm
                    # print(str(ind))

                if flag == 0:
                    flag = 1
                    df_fft['t'] = t

    end = time.time()

    return (df_fft)


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
    dfi_mean = pd.DataFrame()
    dfi_mean[x_col] = dfi[x_col]
    col = dfi.columns[1:]
    dfi_mean[col] = dfi[col].rolling(window=win_size).mean()
    dfi_mean = dfi_mean.add_suffix('_mean')
    dfi_mean = dfi_mean.rename(columns={dfi_mean.columns[0]: x_col})
    # dfi_var[col]=dfi_var.append(dfi_temp)

    end = time.time()

    return (dfi_mean)


# <codecell> STFT Transform Calculation function
def df_stft_RAW_calc(dfx, Fs, fft_win, fft_win_overlap, ch_arr, record_type=''):
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
             f, t, Zxx = signal.stft(sig, Fs, window=w, nperseg=N, noverlap=fft_win_overlap)
    """
    start = time.time()
    Z_arr = []
    Name_arr = []
    print_flag = 0
    for meas_sig in ch_arr:
        N = fft_win  # Number of point in the fft
        w = signal.hamming(N)  # FFT window
        # w = signal.blackman(N) #FFT window
        for col_fft in dfx.columns[:]:
            # f_arr=zero_span_arr
            # if meas_sig in col_fft:
            if meas_sig == col_fft:
                if record_type == 'Orion SPI':
                    # average = dfx[col_fft].mean()
                    volt_per_div = voltage_peak_to_peak / (2 ** adc_bits)
                    # sig = (dfx[col_fft] - average) * volt_per_div
                    sig = dfx[col_fft] * volt_per_div
                else:
                    sig = dfx[col_fft]
                try:
                    f, t, Zxx = signal.stft(sig, Fs, window=w, nperseg=N, noverlap=fft_win_overlap)
                except:
                    f, t, Zxx = signal.stft(sig.astype(float), Fs, window=w, nperseg=N, noverlap=fft_win_overlap)
                Z_arr.append(Zxx)
                Name_arr.append(col_fft)
                del Zxx
                if print_flag == 0:
                    txt = meas_sig + ' FFT Window Time Frame = ' + str(round((N / Fs * 1000), 3)) + ' [mSec]'
                    print_flag = 1
    gc.collect()
    end = time.time()

    return (t, f, Z_arr, Name_arr)


# <codecell> Z Magnitude Calculation function
def Z_mag_calc(Zraw_arr, meas_type='dBW'):
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

    ZdBm_arr = []
    if meas_type == 'Volts PTP to dBm':
        for i in range(len(Zraw_arr)):
            Zxx = np.abs(Zraw_arr[i])
            Zxx = Zxx / (np.sqrt(2) * 2) ** 2   # Vpp to Vrms according to Gil
            # Zxx = Zxx * (1 / np.sqrt(2))
            Zxx = Zxx * 2   # for the FFT
            Zxx = (Zxx / 50) * 1000
            Zxx[Zxx == 0] = 10 ** -12
            Zxx = 10 * np.log10(Zxx)
            ZdBm_arr.append(Zxx)
            del Zxx
            gc.collect()
    else:
        for i in range(len(Zraw_arr)):
            Zxx = Zraw_arr[i]
            Zxx = 2 * np.abs(Zxx)
            Zxx[Zxx == 0] = 10 ** -12
            Zxx = (20 * np.log10(Zxx / 10))
            if meas_type == 'dBW':
                Zxx + 30
            ZdBm_arr.append(Zxx)
            del Zxx
            gc.collect()
    end = time.time()

    return (ZdBm_arr)


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
    Zphase_arr = []
    for i in range(len(Zraw_arr)):
        Zxx = Zraw_arr[i]
        Zxx = np.angle(Zxx, deg=True)
        # Zphase_arr.append(Zxx)
        if phase_unwrap:
            Zxx = np.unwrap(Zxx)
        Zphase_arr.append(Zxx)
        del Zxx
        gc.collect()
    end = time.time()

    return (Zphase_arr)


def scpectrogram_plot(Z, t, f, max_plot_res, fmax, fmin, t_start, t_stop, plot_on, results_path, meas_sig='', ver=1):
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
    start = time.time()
    f_min_ind = int(fmin / (f[1] - f[0]))
    f_max_ind = int(fmax / (f[1] - f[0])) + 1
    spec_res = int(len(t) / max_plot_res)
    freq_res = int(len(f) / max_plot_res)

    if spec_res == 0:
        spec_res = 1
    if freq_res == 0:
        freq_res = 1

    trace = [go.Heatmap(x=t[::spec_res], y=f[f_min_ind:f_max_ind],
                        z=Z[f_min_ind:f_max_ind, ::spec_res], colorscale='Jet')]

    layout = go.Layout(title='Spectrogram [dBm]: ' + results_path.split('\\')[-1] if meas_sig == '' else meas_sig,
                       yaxis=dict(title='Frequency [Hz]'),  # x-axis label
                       xaxis=dict(title='Time [sec]'),  # y-axis label
                       )

    fig = go.Figure(data=trace, layout=layout)
    config = {'scrollZoom': True, 'editable': True}
    if ver == 1:
        if meas_sig == '':
            meas_sig = '01 Spectrogram'
        txt = results_path + '\\' + meas_sig + '.html'
        # fig.write_image(txt + '.jpeg')
        try:
            py.offline.plot(fig, auto_open=plot_on, config=config, filename=txt)
        except:
            print('Something failed, in py.offline.plot (NDF_V2 line 590ish)')
    else:
        txt = results_path + '\\' + meas_sig.replace('"', '')
        # fig.write_image(txt + '.jpeg')
        py.offline.plot(fig, auto_open=plot_on, config=config, filename=txt + ' Spectrogram.html')
    end = time.time()
    print(f'finished Spectrogram; length = {end - start}')


def Avg_no_overlap(arr, N):
    # ARR TO AVERGAE BY N SAMPLES
    # ids = np.arange(len(arr)) // N
    # out = np.bincount(ids, arr) / np.bincount(ids)
    # return(out)
    out = []
    N = int(N)
    for i in range(0, len(arr), N):
        sliced = arr[i:N + i]
        out.append(statistics.mean(sliced))

        if (len(sliced)) < N:
            break
    return np.array(out)


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

    start = time.time()
    if win_size > len(t) - 1:
        print("win size is to long! Please reduce the number of samples to be below " + str(len(t)) + " value")
        return ()

    if win_size < win_overlap:
        print(
            "win overlap size is to long! Please reduce the number of samples to be below " + str(win_size) + " value")
        return ()

    df_MH = pd.DataFrame()
    df_AVG = pd.DataFrame()
    df_MIN = pd.DataFrame()

    # df_MH = pd.DataFrame(columns=['f'])
    # df_AVG = pd.DataFrame(columns=['f'])
    # df_MH['f']=f
    # df_AVG['f']=f

    win_size = int(win_size)
    win_overlap = int(win_overlap)

    # tstep=t[1]-t[0]
    i1 = 0
    i2 = win_size

    di = win_size - win_overlap
    if di == 0:
        di = 1;
    t_list = t[i2::di]

    while i2 < len(t):
        Zt = Z[:, i1:i2]
        Zt_AVG = Zt.mean(axis=1)
        Zt_MH = Zt.max(axis=1)
        if return_min_with_max_avg:
            Zt_MIN = Zt.min(axis=1)
        # Zt_MH=Zt.std(axis=1)

        t1 = t[i1]
        t2 = t[i2]
        col_name = meas_sig + ' @ t = ' + str(round(t1, 3)) + '-' + str(round(t2, 3)) + ' [S]'
        df_MH[col_name] = Zt_MH
        df_AVG[col_name] = Zt_AVG
        if return_min_with_max_avg:
            df_MIN[col_name] = Zt_MIN

        i1 = i1 + di
        i2 = i2 + di

    end = time.time()
    if return_min_with_max_avg:
        return df_MH, df_AVG, t_list, df_MIN
    else:
        return (df_MH, df_AVG, t_list)


# <codecell> Calculates Sliding DWT Coeeficients from data Frame col
def Sliding_WDT(dfx, x_col, data_col, win_size, overlap, waveletname, decomp_level):
    """
        Function calculates WDT coeeficients and exports it as DataFrame

    """
    start = time.time()
    if win_size < overlap: raise ValueError('Window size must be larger than overlap')

    overlap_ratio = overlap / win_size

    N = NextPowerOfTwo(win_size)
    win_size = int(2 ** N)
    overlap = int(overlap_ratio * win_size)
    txt = '\nNormalization to next power of 2 \n ==> new win_size = %i,  new overlap = %i\n' % (win_size, overlap)

    ds = win_size - overlap
    idx_start = 0
    idx_end = int(win_size - 1)
    Fs = round(1 / (dfx[x_col].iloc[1] - dfx[x_col].iloc[0]))
    t_array = dfx[x_col].iloc[idx_end::ds]

    df_wavelet = pd.DataFrame(columns=[x_col, data_col])
    df_wavelet[x_col] = t_array
    l = len(t_array)
    avg_data = np.zeros(l)
    avg_coeffs = np.zeros((decomp_level, l))

    df_col_name = []

    data = dfx[data_col].to_numpy()
    k = 0
    while idx_end < len(dfx):
        data_segment = data[idx_start:idx_end]
        idx_start = idx_start + ds
        idx_end = idx_start + win_size - 1

        seg = np.abs(data_segment)
        seg = np.power(seg, 2)
        avg_data[k] = np.mean(seg)

        coeffs = pywt.wavedec(data_segment, waveletname, mode='zero', level=decomp_level)

        i = decomp_level
        j = 1

        while i > 0:
            di = coeffs[i]
            di = np.abs(di)
            di = np.power(di, 2)
            avg_coeffs[j - 1, k] = np.mean(di)
            if k == 0:
                fi_h_kHz = round((Fs / (2 ** j)) / 1000, 1)
                fi_l_kHz = round((Fs / (2 ** (j + 1))) / 1000, 1)
                if j <= decomp_level:
                    d_col = 'wdt_d' + str(j) + ' @ ' + str(((fi_l_kHz))) + '[kHz]-' + str((fi_h_kHz)) + '[kHz]'
                else:
                    d_col = 'wdt_a' + str(j - 1) + ' @ ' + str(0) + '[kHz] - ' + str(fi_l_kHz) + '[kHz]'
                df_col_name.append(d_col)
            i = i - 1
            j = j + 1

        k = k + 1
    df_wavelet[data_col] = avg_data.tolist()
    for i in range(0, decomp_level):
        df_wavelet[df_col_name[i]] = avg_coeffs[i].tolist()

    end = time.time()
    # df_wavelet=df_wavelet.reset_index()
    return (df_wavelet)


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
            y = samples[n] + w_real * d1 - d2
            d2, d1 = d1, y

        # Storing results `(real part, imag part, power)`
        results.append((
            0.5 * w_real * d1 - d2, w_imag * d1,
            d2 ** 2 + d1 ** 2 - w_real * d1 * d2)
        )
        freqs.append(f * sample_rate)
    return freqs, results


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

    start = time.time()

    # get signal vector and sample rate
    sx = dfx[sig_col_name].to_numpy()

    # get time vector and sample rate
    t = dfx[t_col_name].to_numpy()
    sample_rate = 1 / (t[2] - t[1])

    if win_size > len(t) - 1:
        print("win size is to long! Please reduce the number of samples to be below " + str(len(t)) + " value")
        return ()

    if win_size < win_overlap:
        print(
            "win overlap size is to long! Please reduce the number of samples to be below " + str(win_size) + " value")
        return ()

    # calculate frequency resolution
    f_resolution = sample_rate / win_size
    f1 = FFT_freq + (f_resolution / 2)
    f2 = FFT_freq + (f_resolution)
    results_df = pd.DataFrame(columns=['t', 'data'])

    win_size = int(win_size)
    win_overlap = int(win_overlap)

    i1 = 0
    i2 = win_size
    di = win_size - win_overlap
    t_list = t[i2::di]
    results_df['t'] = t_list
    df_ind = 0
    while i2 < len(t):
        sxt = sx[i1:i2]
        freqs, results = goertzel(sxt, sample_rate, (f1, f2))
        res_P = results[0][2]
        results_df.loc[df_ind, 'data'] = res_P
        df_ind = df_ind + 1
        i1 = i1 + di
        i2 = i2 + di

    col_name = sig_col_name + ' @ f = ' + str(round((freqs[0] / 1000), 2)) + ' [kHz]'
    results_df = results_df.rename(columns={'t': 'Time', 'data': col_name})

    end = time.time()

    return (results_df)


# <codecell> Multiple Pairs ZeroSpan Covariance and Correlation function
def ZeroSpan_Correlator(dfx, t_col_name, win_size, freq_pairs):
    """
        Function calculates covariance and correlation on 2 ZeroSpan results
        Inputs:
            `dfx`               - Input data frame
            `t_col_name`    - name of time vector in data frame  (str)
            `win_size`          - number of samples of covariance/correlation calcs
            `freq_pairs`        - a tuplet of source vector pairs [[30.0,60.0],[70.0, 30.0],[16.6, 50.0],...], each pair is a str series with partial vector names
        Outputs:
            `df_pairs`          - dataframe with time and analysed signals pairs
            `df_corr`           - dataframe with time and corralation results
            `df_cov`            - dataframe with time and covariance results
    """
    df_pairs = pd.DataFrame(columns=[t_col_name])
    df_corr = pd.DataFrame(columns=[t_col_name])
    df_cov = pd.DataFrame(columns=[t_col_name])
    df_pairs[t_col_name] = dfx[t_col_name]
    df_corr[t_col_name] = dfx[t_col_name]
    df_cov[t_col_name] = dfx[t_col_name]

    for fp in freq_pairs:
        for col in dfx.columns:
            if fp[0] in col and 'mean' not in col:
                df_pairs[col] = dfx[col]
                a = col
            if fp[1] in col and 'mean' not in col:
                df_pairs[col] = dfx[col]
                b = col

        c = a + ' vs ' + fp[1] + ' [kHz]'
        df_corr[c] = df_pairs.rolling(window=win_size).corr().unstack()[a][b]
        df_cov[c] = df_pairs.rolling(window=win_size).cov().unstack()[a][b]

    return (df_pairs, df_corr, df_cov)


# <codecell> AFCI_Stage1_Functions
def AFCI_Stage_1(dfx, t_col_name, WindowSize, noise_method, FilterSize, Threshold, K):
    """
        Function calculates covariance and correlation on 2 ZeroSpan results
        Inputs:
            `dfx`           - Input data frame
            `t_col_name`    - name of time vector in data frame  (str)
            `noise_win`     - number of samples of noise filter
            `noise_method`  - string, avg/max/min
            `signal_win`    - number of samples of noise filter

        Outputs:
            `df_SNR`        - SNR dataframe with time and analysed signals

    """
    df_STG1 = pd.DataFrame(columns=[t_col_name])
    df_STG1[t_col_name] = dfx[t_col_name]

    for col in dfx.columns[1:]:
        if 'mean' not in col and t_col_name not in col and 'Sample' not in col:
            SampleIndex = WindowSize + FilterSize + 20  # Skipping the first 20 samples
            signal = dfx[col].to_numpy()
            detection = np.zeros(len(signal))
            while (SampleIndex < len(signal)):
                i = SampleIndex - WindowSize
                if noise_method == 'max':
                    Noise = max(signal[(SampleIndex - WindowSize - FilterSize):i])
                if noise_method == 'min':
                    Noise = min(signal[(SampleIndex - WindowSize - FilterSize):i])
                if noise_method == 'avg':
                    Noise = np.average(signal[(SampleIndex - WindowSize - FilterSize):i])

                OverThresholdCounter = 0
                while (i < SampleIndex):
                    sample = signal[i]
                    if (abs(sample - Noise) > Threshold):
                        OverThresholdCounter += 1
                    if (OverThresholdCounter >= K):
                        detection[i] = 1
                        break
                    i += 1
                SampleIndex += 1

            df_STG1[col] = detection.tolist()
    return (df_STG1)


# <codecell> Find next power of 2
def NextPowerOfTwo(number):
    """
    # Returns next power of two following 'number'
    """
    return np.ceil(np.log2(number))


# <codecell> Extend numpy array to next power of 2 samples
def Extend_to_NextPowerOfTwo(arr, pad_mode='constant'):
    """
    # Returns next power of two following 'number'

    """
    nextPower = NextPowerOfTwo(len(arr))
    deficit = int(math.pow(2, nextPower) - len(arr))
    extended_arr = np.pad(arr, (0, deficit), mode=pad_mode)
    return extended_arr


#######from ndf old ######

def ndf_data_plot(dfx, data_name, x_col, max_plot_res, auto_open_plot, file_on, results_path):
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
    plot_res = int(len(dfx) / max_plot_res)
    if plot_res == 0:
        plot_res = 1
    data_out = [x_col]
    # fig_out=[]

    data = []
    for col in dfx:
        data.append(
            go.Scattergl(
                x=dfx[x_col].iloc[::plot_res],
                y=dfx[col].iloc[::plot_res],
                name=col + ' - ' + data_name,
                mode='lines',
                # mode = 'markers',
                hoverlabel=dict(namelength=-1)
            )
        )

    fig = go.FigureWidget(data)
    config = {'scrollZoom': True, 'editable': True}

    fig['layout'].update(xaxis=dict(title=x_col))
    data_out.append(data)
    # fig_out.append(fig)
    txt = results_path + '/' + ' ' + data_name
    if file_on:
        py.offline.plot(fig, auto_open=auto_open_plot, config=config, filename=txt + '.html')

    end = time.time()

    return (data_out)


def npz_to_array(path):
    data = np.load(path + '/mat.npz')

    return data.f.t, data.f.f, data.f.Zraw


def Save_Zraw(results_path, t, f, Zraw):
    np.savez(results_path + '\mat.npz', t=t, f=f, Zraw=Zraw)
    retur


def Save_df_fft_mag(results_path, df_fft, test_name):
    df_fft.to_csv(results_path + '/' + test_name + ' df_fft_mag.csv', index=False, header=True)


def Save_df_fft_phase(results_path, df_fft, test_name):
    df_fft.to_csv(results_path + '/' + test_name + ' df_fft_phase.csv', index=False, header=True)


def df_dBTOv_calc_all(dfi):
    """
        Function calculates 20*Log10(dfi) dataframe columns
        `dfi`       - input Pandas Data Frame
        `x_col`     - exact x axis col name

    """
    start = time.time()

    dfi_V = pd.DataFrame()
    col = dfi.columns[1:]
    # dfi[dfi==240]=10**-12
    dfi_V[col] = np.exp(dfi[col] / 20)
    # dfi_V=dfi_V.rename(columns={dfi_V.columns[0]: x_col})
    end = time.time()

    return (dfi_V)


def df_dBTOv_calc(dfi, x_col):
    """
        Function calculates 20*Log10(dfi) dataframe columns
        `dfi`       - input Pandas Data Frame
        `x_col`     - exact x axis col name

    """
    start = time.time()

    dfi_V = pd.DataFrame()
    dfi_V[x_col] = dfi[x_col]
    col = dfi.columns[1:]
    # dfi[dfi==240]=10**-12
    dfi_V[col] = np.exp(dfi[col] / 20)
    dfi_V = dfi_V.rename(columns={dfi_V.columns[0]: x_col})
    end = time.time()

    return (dfi_V)


def Read_df_fft(path):
    df_fft = pd.read_csv(path, header=0)  # Reading the df_fft from the  LV450 Basic AFD Evaluation V4.0
    df_fft = df_fft[df_fft['t'] > 0.9]  # removing junk
    return df_fft


def SNR(df_fft, zero_span_arr, results_path, plot, factor, string):
    df_fft = df_fft[df_fft['t'] > 0.9]
    t = df_fft[['t']]
    win_size = 10
    data = []
    SNR = pd.DataFrame()
    for col in df_fft.columns[1:]:
        # plt.plot(t,df_fft[col])
        # plt.show()
        Signal = df_dBTOv_calc(df_fft, col)  # in VOLTS

        Noise = Signal[col].rolling(window=win_size).std().median() / np.sqrt(win_size)
        # print(Noise)
        AVG = Signal[col].rolling(window=win_size).mean().median()
        # print(AVG)
        TEMP = (abs(Signal[col] - AVG)) / Noise
        TEMP[TEMP == 0] = 10 ** -12
        SNR[col] = 20 * np.log10(TEMP)

        SNR[col] = SNR[col].rolling(window=100).mean()
        sp = np.fft.fft(np.sin(t))
        freq = np.fft.fftfreq(t.shape[-1])

        data.append(
            go.Scattergl(
                x=t['t'],
                y=SNR[col],
                mode='lines',
                name=col,
                # mode = 'markers',
                hoverlabel=dict(namelength=-1)
            )
        )
        Signal = []
        Noise = 0
        AVG = 0

    fig = go.FigureWidget(data)
    config = {'scrollZoom': True, 'editable': True}
    fig['layout'].update(title=string + 'SNR')
    fig['layout'].update(xaxis=dict(title='time'))
    fig['layout'].update(yaxis=dict(title='SNR[dB]'))
    # data_out.append(data)
    results_path = results_path + '/' + str(factor)
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    py.offline.plot(fig, auto_open=plot, config=config,
                    filename=results_path + '/' + string + ' SNR ds by ' + str(factor) + ', the df_fft is ' + str(
                        round(zero_span_arr[0] / 1000, 2)) + '[Khz].html')
    return SNR, t['t']


def MH_plot_for_gui(res_name_arr, ZdBm, t, f, MH_time, Overlap_time, name, Factor, ch_arr, plot, results_path):
    t_resolution = 0.001
    win_size = int(MH_time / t_resolution)
    win_overlap = int(Overlap_time / t_resolution)
    max_plot_res = 100000

    indices = [i for i, elem in enumerate(res_name_arr)]

    df_MH_1 = pd.DataFrame(columns=['f'])
    df_AVG_1 = pd.DataFrame(columns=['f'])

    df_MH_1['f'] = f
    df_AVG_1['f'] = f
    for i in indices:
        meas_sig = res_name_arr[i]
        (df_MH_temp, df_AVG_temp, t_list) = sliding_spectrum(ZdBm[i], t, f, win_size, win_overlap, meas_sig)

        df_MH_1[df_MH_temp.columns] = df_MH_temp
        df_AVG_1[df_AVG_temp.columns] = df_AVG_temp

        del df_MH_temp
        del df_AVG_temp
    file_on = False
    MH1_plot_data = data_plot(df_MH_1, 'Sliding Spectrum MH', 'f', max_plot_res, ch_arr[0], plot_on, file_on,
                              results_path)

    data_list = [MH1_plot_data]

    name_list = ['Sliding MH Spectrum dBm']
    x_sync = True
    data_name = str(Factor) + name + ' Sliding FFT MH'
    data_pane_plot(data_name, data_list, name_list, plot_on, x_sync, ch_arr, ch_arr[0], results_path)


def MH_plot(res_name_arr, ZdBm, t, f, MH_time, Overlap_time, name, Factor, ch_arr, plot, results_path):
    t_resolution = 0.001
    win_size = int(MH_time / t_resolution)
    win_overlap = int(Overlap_time / t_resolution)
    max_plot_res = 100000

    indices = [i for i, elem in enumerate(res_name_arr)]

    df_MH_1 = pd.DataFrame(columns=['f'])
    df_AVG_1 = pd.DataFrame(columns=['f'])

    df_MH_1['f'] = f
    df_AVG_1['f'] = f
    for i in indices:
        meas_sig = res_name_arr[i]
        (df_MH_temp, df_AVG_temp, t_list) = sliding_spectrum(ZdBm[i], t, f, win_size, win_overlap, meas_sig)

        df_MH_1[df_MH_temp.columns] = df_MH_temp
        df_AVG_1[df_AVG_temp.columns] = df_AVG_temp

        del df_MH_temp
        del df_AVG_temp
    file_on = False
    MH1_plot_data = data_plot(df_MH_1, 'Sliding Spectrum MH', 'f', max_plot_res, ch_arr[0], plot_on, file_on,
                              results_path)

    tag_arr = ['VRX']
    data_list = [MH1_plot_data]

    name_list = ['Sliding MH Spectrum dBm']
    x_sync = True
    data_name = str(Factor) + ' ' + name + ' Sliding FFT MH and AVG Spectrum'
    data_pane_plot(data_name, data_list, name_list, plot_on, x_sync, tag_arr, ch_arr[0], results_path)


def SNR_plus(df_fft, zero_span_arr, results_path, plot, factor, string):
    df_fft = df_fft[df_fft['t'] > 0.9]
    t = df_fft[['t']]
    win_size = 10
    data = []
    data_out = [t]
    SNR = pd.DataFrame()
    Signal_in_time = pd.DataFrame()
    Signal = df_dBTOv_calc_all(df_fft)
    for col in df_fft.columns[1:]:
        for one_freq in zero_span_arr:
            if str(one_freq) in col:
                # plt.plot(t,df_fft[col])
                # plt.show()
                # Signal=df_dBTOv_calc(df_fft,col)# in VOLTS
                window = signal.gaussian(10, 5)
                plt.plot(Signal[col])
                plt.show()
                Signal_in_time[col] = np.convolve(Signal[col], window)
                # Signal=df_dBTOv_calc(df_fft,col)# in VOLTS
                plt.plot(Signal_in_time)
                plt.show()

                Noise = Signal_in_time[col].rolling(window=win_size).std().median() / np.sqrt(win_size)
                AVG = Signal_in_time[col].rolling(window=win_size).mean().median()
                TEMP = ((abs(Signal_in_time[col] - AVG)) / Noise)
                TEMP[TEMP == 0] = 10 ** -12
                SNR[col] = 20 * np.log10(TEMP)

                data.append(
                    go.Scattergl(
                        x=t['t'],
                        y=SNR[col],
                        mode='lines',
                        name=col,
                        # mode = 'markers',
                        hoverlabel=dict(namelength=-1)
                    )
                )
                Signal_in_time = []
                Noise = 0
                AVG = 0

    fig = go.FigureWidget(data)
    config = {'scrollZoom': True, 'editable': True}
    # fig['layout'].update(title=name)
    # fig['layout'].update(xaxis=dict(title = t))
    data_out.append(data)

    py.offline.plot(fig, auto_open=True, config=config, filename=results_path + '/' + 'df_fft.html')
    return data_out


def SNR_Matced(df_fft, zero_span_arr, results_path, plot, factor, string):
    df_fft = df_fft[df_fft['t'] > 0.9]
    t = df_fft[['t']]
    win_size = 10
    data = []
    data_out = [t]
    Signal_in_time = pd.DataFrame()
    SNR = pd.DataFrame()
    for col in df_fft.columns[1:]:
        # plt.plot(t,df_fft[col])
        # plt.show()
        Signal = df_dBTOv_calc(df_fft, col)  # in VOLTS
        window = signal.gaussian(10, 0.05)
        # plt.plot(Signal[col])
        # plt.show()
        Signal_in_time[col] = np.convolve(Signal[col], window)
        Noise = Signal[col].rolling(window=win_size).std().median() / np.sqrt(win_size)
        AVG = Signal[col].rolling(window=win_size).mean().median()
        TEMP = (abs(Signal_in_time[col] - AVG)) / Noise
        TEMP[TEMP == 0] = 10 ** -12
        SNR[col] = 20 * np.log10(TEMP)
        SNR[col]

        sp = np.fft.fft(np.sin(t))
        freq = np.fft.fftfreq(t.shape[-1])

        data.append(
            go.Scattergl(
                x=t['t'],
                y=SNR[col],
                mode='lines',
                name=col,
                # mode = 'markers',
                hoverlabel=dict(namelength=-1)
            )
        )
        Signal = []
        Noise = 0
        AVG = 0

    fig = go.FigureWidget(data)
    config = {'scrollZoom': True, 'editable': True}
    fig['layout'].update(title=string + 'SNR')
    fig['layout'].update(xaxis=dict(title='time'))
    fig['layout'].update(yaxis=dict(title='SNR[dB]'))
    # data_out.append(data)
    results_path = results_path + '/' + str(factor)
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    py.offline.plot(fig, auto_open=plot, config=config,
                    filename=results_path + '/' + string + ' SNR ds by ' + str(factor) + ', the df_fft is ' + str(
                        round(zero_span_arr[0] / 1000, 2)) + '[Khz].html')
    return SNR, t['t']


def Had_telem(df_fft, zero_span_arr, results_path, win_size):
    df_fft = df_fft[df_fft['t'] > 1]
    t = df_fft[['t']]
    df_xcor = pd.DataFrame(columns=['t'])
    df_xcor['t'] = df_fft['t']
    win_size = 10
    data_snr = []
    data_out_snr = [t]
    data = []
    data_out = [t]
    SNR = pd.DataFrame()
    DIFF = pd.DataFrame()
    for col in df_fft.columns[1:]:
        for one_freq in zero_span_arr:
            if str(one_freq) in col:
                # plt.plot(t,df_fft[col])
                # plt.show()
                Signal = df_dBTOv_calc(df_fft, col)  # in VOLTS
                Signal.to_csv(results_path + '/' + ' signal.csv', index=False, header=True)
                Noise = Signal[col].rolling(window=win_size).std().median() / np.sqrt(win_size)
                AVG = Signal[col].rolling(window=win_size).mean().median()
                TEMP = (abs(Signal[col] - AVG)) / Noise
                TEMP.to_csv(results_path + '/' + ' temp.csv', index=False, header=True)
                TEMP[TEMP == 0] = 10 ** -12
                SNR[col] = 20 * np.log10(TEMP)
                SNR.to_csv(results_path + '/' + ' snr.csv')
                AVG_SNR = SNR[col].rolling(window=win_size).mean().median()
                TOP_SNR = AVG_SNR * 2
                DIFF = SNR[col]

                len_of_sig = (len(DIFF))
                i = 0
                while i < len_of_sig:  # using the snr defien where is the telem!
                    temp = DIFF[i:i + 5].values.tolist()
                    res = all(i < j for i, j in zip(temp, temp[1:]))
                    if res:
                        sample = DIFF.iloc[i]
                        if sample > TOP_SNR:
                            # print (i)
                            is_telem = statistics.mean(DIFF[i:i + 200].values.tolist())
                            if is_telem > 35:  # print('this is telem')
                                i = i + 300

                    i += 1
                # Signal=[]
                Noise = 0
                AVG = 0

    # df = pd.DataFrame(Signal)
    SNRK = SNR.rolling(window=500).mean()
    # COV=Signal.rolling(window=win_size).cov().unstack()#.to_csv(results_path+'/' +' COV1.csv')
    COV = Signal.rolling(window=win_size).cov().unstack()  # .to_csv(results_path+'/' +' COV1.csv')
    # NUMPT_COV=COV.to_numpy()
    # for col in COV:
    #     tempt=COV[col]

    for col in COV.columns:
        data.append(
            go.Scattergl(
                x=t['t'],
                y=COV[col],
                mode='lines',
                name=col[0] + col[1],
                # mode = 'markers',
                hoverlabel=dict(namelength=-1)
            )
        )

    fig = go.FigureWidget(data)
    config = {'scrollZoom': True, 'editable': True}
    # fig['layout'].update(title=name)
    # fig['layout'].update(xaxis=dict(title = t))
    data_out.append(data)
    py.offline.plot(fig, auto_open=True, config=config, filename=results_path + '/' + 'COV_fft.html')

    # py.offline.plot(fig,auto_open = False, config=config, filename=results_path+'/'+'df_fft.html')
    return data_out, COV


def stage_1_energy_raise(EnergyDB, WindowSize, FilterSize, OverThresholdLimit):
    EnergyThresholdList = []
    EnergyThresholdList.append(0)
    SampleIndex = WindowSize + FilterSize + 20  # Skipping the first 20 samples because of Inverter noises.
    while (SampleIndex < len(EnergyDB)):
        EnergyThreshold = 4;
        MinFilterWindow = min(EnergyDB[(SampleIndex - WindowSize - FilterSize):SampleIndex - WindowSize])
        while EnergyThreshold < 50:
            OverThresholdCounter = 0
            i = SampleIndex - WindowSize
            while (i < SampleIndex):

                if EnergyDB.iloc[i] > (MinFilterWindow + EnergyThreshold + 1):
                    # if MinFilterWindow-EnergyDB.iloc[i]>(EnergyThreshold+1):
                    OverThresholdCounter += 1
                if (OverThresholdCounter >= OverThresholdLimit):
                    break
                i += 1
            if (OverThresholdCounter < OverThresholdLimit):
                break
            EnergyThreshold += 0.5
        if EnergyThreshold == 4:
            EnergyThreshold = 0
        else:
            EnergyThresholdList.append(EnergyThreshold)
        SampleIndex += 1
    print(W + 'The max energy raise is ' + P + str(max(EnergyThresholdList)))
    return max(EnergyThresholdList)


def stage_1_Iac_raise(Iac_arr, WindowSize, FilterSize, OverThresholdLimit):
    Iac_ThresholdList = []
    Iac_ThresholdList.append(0)
    SampleIndex = WindowSize + FilterSize + 20  # Skipping the first 20 samples because of Inverter noises.
    while (SampleIndex < len(Iac_arr)):
        MaxCurrentDrop = 0.1;
        MinFilterWindow = (Iac_arr[(SampleIndex - WindowSize - FilterSize):SampleIndex - WindowSize]).mean().iloc[0]
        while MaxCurrentDrop < 1:
            OverThresholdCounter = 0
            i = SampleIndex - WindowSize
            while (i < SampleIndex):

                if MinFilterWindow - Iac_arr.iloc[i].values[0] > (+MaxCurrentDrop):
                    # if MinFilterWindow-Iac_arr.iloc[i]>(EnergyThreshold+1):
                    OverThresholdCounter += 1
                if (OverThresholdCounter >= OverThresholdLimit):
                    break
                i += 1
            if (OverThresholdCounter < OverThresholdLimit):
                break
            MaxCurrentDrop += 0.05
        if MaxCurrentDrop == 0.1:
            MaxCurrentDrop = 0
        else:
            Iac_ThresholdList.append(MaxCurrentDrop)
        SampleIndex += 1
    print(W + 'The max jittring in the Iac is ' + R + str(max(Iac_ThresholdList)))
    return max(Iac_ThresholdList)


def cheak_harmonics(f_resampling, fpeak, f_fft, k, good_k):
    for i in range(1, k + 1):
        if (abs(f_fft + 1000) < abs(f_resampling - i * fpeak) or abs(f_resampling - i * fpeak) < abs(f_fft - 1000)):
            good_k += 1
    return good_k


def Get_downsampled_signal(x, fs, target_fs, order, Lpf_type):
    decimation_ratio = np.round(fs / target_fs)
    if fs < target_fs:
        raise ValueError("Get_downsampled_signal")
    else:
        try:
            if Lpf_type == 'None':
                y0 = x[::int(decimation_ratio)]
            else:
                y0 = signal.decimate(x, int(decimation_ratio), order, zero_phase=True, ftype=Lpf_type)
            # y1 = signal.decimate(y0,2, 2,zero_phase=True,ftype='iir')
            # f_poly = signal.resample_poly(y, 100, 20)
        except:
            print('error in ds func!')
        actual_fs = fs / decimation_ratio
    return y0, actual_fs


def plot_wavelet(ax, time2, signal, scales, waveletname='cmor',
                 cmap=plt.cm.seismic, title='', ylabel='', xlabel=''):
    dt = time2
    coefficients, frequencies = pywt.cwt(signal, scales, waveletname, dt)

    power = (abs(coefficients)) ** 2
    period = frequencies
    levels = [0.015625, 0.03125, 0.0625, 0.125, 0.25, 0.5, 1]
    contourlevels = np.log2(levels)  # original
    time = range(2048)

    im = ax.contourf(time, np.log2(period), np.log2(power), contourlevels, extend='both', cmap=cmap)

    ax.set_title(title, fontsize=20)
    ax.set_ylabel(ylabel, fontsize=18)
    ax.set_xlabel(xlabel, fontsize=18)
    yticks = 2 ** np.arange(np.ceil(np.log2(period.min())), np.ceil(np.log2(period.max())))
    ax.set_yticks(np.log2(yticks))  # original
    ax.set_yticklabels(yticks)  # original
    ax.invert_yaxis()
    ylim = ax.get_ylim()

    cbar_ax = fig.add_axes([0.95, 0.5, 0.03, 0.25])
    fig.colorbar(im, cax=cbar_ax, orientation="vertical")

    return yticks, yli


def Get_downsampled_signal_NO_FILTER(x, fs, target_fs):
    decimation_ratio = np.round(fs / target_fs)
    if fs < target_fs:
        raise ValueError("Get_downsampled_signal")
    else:
        try:
            y0 = x[::int(decimation_ratio)]
            # y1 = signal.decimate(y0,2, 2,zero_phase=True,ftype='iir')
            # f_poly = signal.resample_poly(y, 100, 20)
        except:
            y0 = x[::int(decimation_ratio)]
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
    Ts = 1 / spi_sample_rate
    Fs = spi_sample_rate
    df_spi = pd.DataFrame(columns=['Time'])
    df = pd.read_csv(inputFile, header=0)
    i = 0
    for col in df.columns:
        if 'Unnamed' in col:
            # print(col)
            del df[col]
        else:
            df = df.rename(columns={col: tag_arr[i]})
            i = i + 1

    df = df.add_prefix(tag_arr[0] + '_')
    df_spi['Time'] = (df.index) * Ts
    # for col in df.columns:
    #     if 'V' in col:
    #         V_quantization=1/(2**6)
    #         df_spi[col]=df*V_quantization
    #     if 'I' in col:
    #         V_quantization=1/(2**9)
    #         df_spi[col]=df*V_quantization
    #     else:
    #         V_quantization=1/(2**12)
    #         df_spi[col]=df*V_quantization

    V_quantization = 1 / (2 ** 6)
    df_spi[df.columns] = df * V_quantization

    df_len = len(df_spi)
    df_time_len = max(df_spi['Time']) - min(df_spi['Time'])
    tmin = min(df_spi['Time'])
    tmax = max(df_spi['Time'])

    temp1 = 'DF Tmin = ' + str(tmin) + '[Sec]; ' + 'DF Tmax = ' + str(tmax) + '[Sec]; \n'
    temp2 = 'DF time length = ' + str(round(df_time_len, 5)) + '[Sec] / ~' + str(
        round(df_time_len / 60, 4)) + '[Min]; \n'
    text = temp1 + temp2 + 'DF length = ' + str(df_len / 1000000) + '[Mega Samples];\n' + 'DF Sampling rate = ' + str(
        round((Fs / 1000), 0)) + '[kSamp/sec]' + '; DF Sampling Interval = ' + str(round((Ts * 1000), 3)) + '[mSec]'

    teee = time.time()

    return (df_spi)


def sliding_spectrum_OLD(dfx, t, f, win_size, win_overlap, meas_sig):
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

    start = time.time()
    if win_size > len(t) - 1:
        print("win size is to long! Please reduce the number of samples to be below " + str(len(t)) + " value")
        return ()

    if win_size < win_overlap:
        print(
            "win overlap size is to long! Please reduce the number of samples to be below " + str(win_size) + " value")
        return ()

    df_MH = pd.DataFrame()
    # df_AVG = pd.DataFrame()

    # df_MH = pd.DataFrame(columns=['f'])
    # df_AVG = pd.DataFrame(columns=['f'])
    # df_MH['f']=f
    # df_AVG['f']=f

    win_size = int(win_size)
    win_overlap = int(win_overlap)

    # tstep=t[1]-t[0]
    i1 = 0
    i2 = win_size
    di = win_size - win_overlap
    t_list = t[i2::di]

    while i2 < len(t):
        Zt = Z[:, i1:i2]
        Zt_AVG = Zt.mean(axis=1)
        Zt_MH = Zt.max(axis=1)
        # Zt_MH=Zt.std(axis=1)
        t1 = t[i1]
        t2 = t[i2]
        col_name = meas_sig + ' @ t = ' + str(round(t1, 3)) + '-' + str(round(t2, 3)) + ' [S]'
        df_MH[col_name] = Zt_MH
        df_AVG[col_name] = Zt_AVG
        i1 = i1 + di
        i2 = i2 + di
    end = time.time()
    return df_MH, df_AVG, t_list


def plot_all(log_energy, window_size, filter_size, over_th_limit):
    energy_th_increment = 1
    energy_th_list = [0 for x in range(window_size + filter_size)]
    sample_index = window_size + filter_size
    while sample_index < len(log_energy):
        min_filter_window = min(log_energy[(sample_index - window_size - filter_size):sample_index - window_size])
        energy_th = 0
        highest_th_is_reached = False
        while not highest_th_is_reached:
            i = sample_index - window_size
            over_threshold_counter = 0
            while i < sample_index:
                if log_energy[i] > min_filter_window + energy_th:
                    over_threshold_counter += 1
                if over_threshold_counter >= over_th_limit:
                    energy_th = energy_th + energy_th_increment
                    break
                if i == sample_index - 1:
                    highest_th_is_reached = True
                    break
                i += 1
        energy_th_list.append(energy_th)
        sample_index += 1
    return max(energy_th_list)


threadLock = threading.Lock()  # define the mutex


def energy_rise(mixer, directory, file, freq, log_energy):
    energy_th_increment = 1
    window_size = 20
    filter_size = 15
    over_th_limit = 12
    energy_th_list = [0 for x in range(window_size + filter_size)]
    sample_index = window_size + filter_size
    while sample_index < len(log_energy):
        min_filter_window = min(log_energy[(sample_index - window_size - filter_size):sample_index - window_size])
        energy_th = 0
        highest_th_is_reached = False
        while not highest_th_is_reached:
            i = sample_index - window_size
            over_threshold_counter = 0
            while i < sample_index:
                if log_energy[i] > min_filter_window + energy_th:
                    over_threshold_counter += 1
                if over_threshold_counter >= over_th_limit:
                    energy_th = energy_th + energy_th_increment
                    break
                if i == sample_index - 1:
                    highest_th_is_reached = True
                    break
                i += 1
        energy_th_list.append(energy_th)
        sample_index += 1
    threadLock.acquire()
    print(f'{mixer},{directory},{file},{freq},{max(energy_th_list)}')
    threadLock.release()
