#write spectogram to first coulm in a dataframe
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import NDF_V4
import scipy.signal as signal
import plotly.graph_objs as go
import os
import glob
import plotly.express as px
import time
def stft(dataframe,ZdBm,f,t,f_resolution,t_resolution,max_plot_res):
    channels = dataframe.loc[:, ~dataframe.columns.str.contains('Time')].columns
    MH_data_out = []
    Avg_data_out = []
    fig_STFT_AVG = NDF_V4.make_fig(row=1, col=1, subplot_titles=channels, shared_xaxes=True)
    fig_STFT_MH = NDF_V4.make_fig(row=1, col=1, subplot_titles=channels, shared_xaxes=True)

    for channel in dataframe.loc[:, ~dataframe.columns.str.contains('Time')].columns:
        df_MH_1 = pd.DataFrame(columns=['f'])
        df_AVG_1 = pd.DataFrame(columns=['f'])

        df_MH_1['f'] = f
        df_AVG_1['f'] = f
        MH_time = 0.1
        Overlap_time = 0.05

        win_size = int(MH_time / t_resolution)
        win_overlap = int(Overlap_time / t_resolution)
        res_name_arr=dataframe.loc[:, ~dataframe.columns.str.contains('Time')].columns
        indices = [i for i, elem in enumerate(res_name_arr) if channel in elem]
        plot_res = int(len(dataframe) / max_plot_res)
        if plot_res == 0:
            plot_res = 1
        if not indices:
            continue
        for i in indices:
            meas_sig = res_name_arr[i]
            if channel in meas_sig:
                (df_MH_temp, df_AVG_temp, t_list, df_MIN_temp) = NDF_V4.sliding_spectrum(ZdBm[i], t, f,
                                                                                           win_size,
                                                                                           win_overlap,
                                                                                           meas_sig)

            df_MH_1[df_MH_temp.columns] = df_MH_temp.copy()
            df_AVG_1[df_AVG_temp.columns] = df_AVG_temp.copy()
        del df_MH_temp
        del df_AVG_temp
        MH_fig, MH_temp_data = NDF_V4.data_plot_streamlit(df_MH_1, channel + ' Max Hold STFT Data', 'f', max_plot_res)
        Avg_fig, Avg_temp_data = NDF_V4.data_plot_streamlit(df_AVG_1, channel + ' Avg Hold STFT Data', 'f',
                                                              max_plot_res)
        #

        temp = 0
        for col in Avg_temp_data[1]:
            fig_STFT_AVG.add_trace(go.Scattergl(x=col.x[::plot_res],
                                                y=col.y[::plot_res],
                                                name=col.name + " ",
                                                visible=False,
                                                showlegend=True))
            temp += 1;
        plots_per_pane_Avg = (temp)
        for col in MH_temp_data[1]:
            fig_STFT_MH.add_trace(go.Scattergl(x=col.x[::plot_res],
                                               y=col.y[::plot_res],
                                               name=col.name + " ",
                                               visible=False,
                                               showlegend=True))

        plots_per_pane_MH = temp

    for i in range(plots_per_pane_Avg):
        fig_STFT_AVG.data[i].visible = True
    steps = []
    for i in range(0, int(len(fig_STFT_AVG.data) / plots_per_pane_Avg)):
        Temp = channels[i]

        step = dict(
            method="update",
            args=[{"visible": [False] * len(fig_STFT_AVG.data)},
                  {"title": "Slider  switched to step "}],
            label=str(Temp)  # layout attribute
        )
        j = i * plots_per_pane_Avg
        for k in range(plots_per_pane_Avg):
            step["args"][0]["visible"][j + k] = True
        steps.append(step)
    sliders = [dict(
        active=10,
        currentvalue={"prefix": "REC: "},
        pad={"t": 50},
        steps=steps
    )]
    fig_STFT_AVG.update_layout(
        sliders=sliders)





    for i in range(plots_per_pane_MH):
        fig_STFT_MH.data[i].visible = True
    steps = []

    for i in range(0, int(len(fig_STFT_AVG.data) / plots_per_pane_MH)):
        Temp = channels[i]

        step = dict(
            method="update",
            args=[{"visible": [False] * len(fig_STFT_AVG.data)},
                  {"title": "Slider  switched to step "}],
            label=str(Temp)  # layout attribute
        )
        j = i * plots_per_pane_MH
        for k in range(plots_per_pane_MH):
            step["args"][0]["visible"][j + k] = True
        steps.append(step)
    sliders = [dict(
        active=10,
        currentvalue={"prefix": "REC: "},
        pad={"t": 50},
        steps=steps
    )]
    fig_STFT_MH.update_layout(
        sliders=sliders,
)



    return fig_STFT_AVG, fig_STFT_MH
def stft2(dataframe,ZdBm,f,t,f_resolution,t_resolution,max_plot_res,meas_sig):
    channels = dataframe.loc[:, ~dataframe.columns.str.contains('Time')].columns
    MH_data_out = []
    Avg_data_out = []
    fig_STFT_AVG = NDF_V4.make_fig(row=1, col=1, subplot_titles=channels, shared_xaxes=True)
    fig_STFT_MH = NDF_V4.make_fig(row=1, col=1, subplot_titles=channels, shared_xaxes=True)
    MH_time = 0.1
    Overlap_time = 0.05

    win_size = int(MH_time / t_resolution)
    win_overlap = int(Overlap_time / t_resolution)
    plot_res = int(len(dataframe) / max_plot_res)
    for channel in channels:
        df_MH_1 = pd.DataFrame(columns=['f'])
        df_AVG_1 = pd.DataFrame(columns=['f'])

        df_MH_1['f'] = f
        df_AVG_1['f'] = f
        if plot_res == 0:
            plot_res = 1

        if channel in meas_sig:
            i=np.where(channel==channels)[0][0]
            (df_MH_temp, df_AVG_temp, t_list, df_MIN_temp) = NDF_V4.sliding_spectrum(ZdBm[i], t, f,
                                                                                       win_size,
                                                                                       win_overlap,
                                                                                       meas_sig[i])

        df_MH_1[df_MH_temp.columns] = df_MH_temp.copy()
        df_AVG_1[df_AVG_temp.columns] = df_AVG_temp.copy()
        del df_MH_temp
        del df_AVG_temp
        MH_fig, MH_temp_data = NDF_V4.data_plot_streamlit(df_MH_1, channel + ' Max Hold STFT Data', 'f', max_plot_res)
        Avg_fig, Avg_temp_data = NDF_V4.data_plot_streamlit(df_AVG_1, channel + ' Avg Hold STFT Data', 'f',
                                                              max_plot_res)
        #

        temp = 0
        for col in Avg_temp_data[1]:
            fig_STFT_AVG.add_trace(go.Scattergl(x=col.x[::plot_res],
                                                y=col.y[::plot_res],
                                                name=col.name + " ",
                                                visible=False,
                                                showlegend=True))
            temp += 1;
        plots_per_pane_Avg = (temp)
        for col in MH_temp_data[1]:
            fig_STFT_MH.add_trace(go.Scattergl(x=col.x[::plot_res],
                                               y=col.y[::plot_res],
                                               name=col.name + " ",
                                               visible=False,
                                               showlegend=True))

        plots_per_pane_MH = temp

    for i in range(plots_per_pane_Avg):
        fig_STFT_AVG.data[i].visible = True
    steps = []
    for i in range(0, int(len(fig_STFT_AVG.data) / plots_per_pane_Avg)):
        Temp = channels[i]

        step = dict(
            method="update",
            args=[{"visible": [False] * len(fig_STFT_AVG.data)},
                  {"title": "Slider  switched to step "}],
            label=str(Temp)  # layout attribute
        )
        j = i * plots_per_pane_Avg
        for k in range(plots_per_pane_Avg):
            step["args"][0]["visible"][j + k] = True
        steps.append(step)
    sliders = [dict(
        active=10,
        currentvalue={"prefix": "REC: "},
        pad={"t": 50},
        steps=steps
    )]
    fig_STFT_AVG.update_layout(
        sliders=sliders)





    for i in range(plots_per_pane_MH):
        fig_STFT_MH.data[i].visible = True
    steps = []

    for i in range(0, int(len(fig_STFT_AVG.data) / plots_per_pane_MH)):
        Temp = channels[i]

        step = dict(
            method="update",
            args=[{"visible": [False] * len(fig_STFT_AVG.data)},
                  {"title": "Slider  switched to step "}],
            label=str(Temp)  # layout attribute
        )
        j = i * plots_per_pane_MH
        for k in range(plots_per_pane_MH):
            step["args"][0]["visible"][j + k] = True
        steps.append(step)
    sliders = [dict(
        active=10,
        currentvalue={"prefix": "REC: "},
        pad={"t": 50},
        steps=steps
    )]
    fig_STFT_MH.update_layout(
        sliders=sliders,
)



    return fig_STFT_AVG, fig_STFT_MH


def ZeroSpan_calc2(Z_arr, Z_name_arr, t, f, zero_span_arr, meas_sig,df):
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
    channels = df.loc[:, ~df.columns.str.contains('Time')].columns
    for channel in channels:
        flag = 0
        f_arr = zero_span_arr
        z_ind = 0
        if channel in meas_sig:
            i=np.where(channel==channels)[0][0]

        for col_fft in Z_name_arr:
            Z_dBm = Z_arr[z_ind]
            z_ind = z_ind + 1
            if meas_sig in col_fft:
                for ff in f_arr:
                    ind=np.where(f==ff)[0][0]
                   # ind = int(ff / (f[1] - f[0]))
                    z_dBm = (Z_dBm[ind, :])
                    df_fft[col_fft + ' @ ' + str(round(float(f[ind] / 1000), 1)) + ' [kHz]'] = z_dBm
                    # print(str(ind))

                if flag == 0:
                    flag = 1
                    df_fft['t'] = t

    end = time.time()

    return (df_fft)

def linoy_specogram(data, Fs, f_resolution, t_resolution):

    fft_win = int(Fs / f_resolution)
    fft_win_overlap = int(fft_win - (t_resolution * Fs))
    if fft_win > len(df):
        fft_win = int(len(df) / 10)
        fft_win_overlap = int(fft_win * 0.99)

    vector = np.vectorize(np.float64)
    x = vector(data)

    N = fft_win

    w = signal.hamming(N)

    freqs, bins, Pxx = signal.spectrogram(x, Fs, window=w, nfft=N, noverlap=fft_win_overlap, scaling='spectrum',
                                          mode='magnitude')
    fig = go.Figure()
    fig.add_trace(go.Heatmap(z=Pxx,
                             y=freqs,
                             x=bins,
                             # y_label='frequency',
                             visible=True,
                             colorscale='Rainbow',
                             showscale=True,
                             showlegend=False))

    fig.write_html('stam23.html', auto_open=True)
    return fig

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
                    ind=np.where(f==ff)[0][0]
                   # ind = int(ff / (f[1] - f[0]))
                    z_dBm = (Z_dBm[ind, :])
                    df_fft[col_fft + ' @ ' + str(round(float(f[ind] / 1000), 1)) + ' [kHz]'] = z_dBm
                    # print(str(ind))

                if flag == 0:
                    flag = 1
                    df_fft['t'] = t

    end = time.time()

    return (df_fft)

def Noam_specogram(df, Fs, f_resolution, t_resolution,meas_sig,max_plot_res,fmin,fmax):

    fft_win = int(Fs / f_resolution)
    fft_win_overlap = int(fft_win - (t_resolution * Fs))
    if fft_win > len(df):
        fft_win = int(len(df) / 10)
        fft_win_overlap = int(fft_win * 0.99)
    res_name_arr = []
    (t, f, Zraw, res_name_arr) = NDF_V4.df_stft_RAW_calc(df, Fs, fft_win, fft_win_overlap,
                                                         df.loc[:, ~df.columns.str.contains('Time')].columns)
    ZdBm = NDF_V4.Z_mag_calc(Zraw)
    Temp_fig_arr=[]
    trace_arr=[]

    for i in range(len(ZdBm)):
        Temp_fig, trace = NDF_V4.spectrogram_plot_for_streamlit(ZdBm[i], t, f, max_plot_res, fmax, fmin, meas_sig[i])
        trace_arr.append(trace)
        Temp_fig_arr.append(Temp_fig)

    return Temp_fig_arr,trace_arr,Zraw,f,t
def energy_rise_algo2(log_energy, window_size=20, filter_size=15, over_th_limit=12):
    energy_th_list = [0 for x in range(window_size + filter_size)]
    for sample_index in range(window_size + filter_size, len(log_energy)):
        min_filter_window = min(log_energy[(sample_index - window_size - filter_size):sample_index - window_size])
        sorted_window = list(log_energy[(sample_index - window_size):sample_index].sort_values(ascending=False))
        if sorted_window[over_th_limit - 1] - min_filter_window > 0:
            energy_th_list.append(sorted_window[over_th_limit - 1] - min_filter_window)
        else:
            energy_th_list.append(0)
    return energy_th_list
def fix_time_axis(df,Fs):
    if 'Time'.upper() not in [x.upper() for x in df.columns]:
        df['Time'] = df.index * 1 / Fs
    else:
        # find which column is time
        for col in df.columns:
            if 'Time'.upper() in col.upper():
                df = df.rename({col: 'Time'}, axis=1)  # new method

                df.Time = df.Time - df.Time.min()

    return df
def slice_freq(Zraw,f, fmin, fmax):
    f_min_ind = int(fmin / (f[1] - f[0]))
    f_max_ind = int(fmax / (f[1] - f[0])) + 1
    f=f[f_min_ind:f_max_ind]
    for i in range(len(Zraw)):
        Zraw[i]=Zraw[i][f_min_ind:f_max_ind]

    return Zraw,f
def plot_specturm(df):
    df = df.rename(columns={df.columns[0]: 'Freq'})
    # rename the second column to max hold
    df = df.rename(columns={df.columns[1]: 'Max Hold'})
    # rename the third column to average
    df = df.rename(columns={df.columns[2]: 'Average'})
    # remove unnamed columns
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    # plot the data, in x axis put the freq column and in y axis put the max hold column and the average column
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Freq'], y=df['Max Hold'],
                             mode='lines',
                             name='Max Hold'))
    fig.add_trace(go.Scatter(x=df['Freq'], y=df['Average'],
                             mode='lines',
                             name='Average'))
    # UPDTAE THE TITLE OF THE PLOT
    fig.update_layout(title='SPECTRUM-SESTI')
    fig.write_html('sesti SPECTRUM.html', auto_open=True)
def main_old():
    df = pd.read_csv(r"C:\Users\noam.d\OneDrive - SolarEdge\Desktop\spidebug\_Record_20230321_145741.csv")
    # get only the forst cokum in the dataframe
    df = df['ADC_IN0_RX1(1)']-8200
    df=df.to_frame()
    data_pwr = pd.read_csv(r"C:\Users\noam.d\OneDrive - SolarEdge\Desktop\spidebug\_Record_20230321_145741 pwr.txt")
    data_pwr=10*np.log10(data_pwr.iloc[:, 2])
    energy_raise=energy_rise_algo2(data_pwr, window_size=20, filter_size=15, over_th_limit=12)
    energy_raise_x_axis = np.arange(0, len(energy_raise), 1)
    #get the index of data_pwr
    data_pwr_x_axis = np.arange(0, len(data_pwr), 1)

    #plot using plotly
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data_pwr_x_axis,y=data_pwr,
                        mode='lines',
                        name='Energy'))
    fig.add_trace(go.Scatter(x=energy_raise_x_axis,y=energy_raise,
                        mode='lines',
                        name='Energy raise'))
    fig.write_html('Pwr.html', auto_open=True)
    Fs = 50e3/4
    f_resolution = 100
    t_resolution = 1e-3
    max_plot_res=100000000

    # fig_linoy_spectogram=linoy_specogram(data, Fs, f_resolution, t_resolution)
    # fig_linoy_spectogram.write_html('data.html', auto_open=True)
    meas_sig='ADC RX1'
    fig_Noam_spectogram,trace_noam,Zraw,f,t= Noam_specogram(df, Fs, f_resolution, t_resolution, meas_sig,max_plot_res)
    fig_Noam_spectogram.write_html('fig_Noam_spectogram.html', auto_open=True)

    ZdBm = NDF_V4.Z_mag_calc(Zraw)
    # fig_STFT_AVG,fig_STFT_MH=stft(df,ZdBm,f,t,f_resolution,t_resolution,max_plot_res)
    #
    #
    # fig_STFT_AVG.write_html('STFT_AVG.html', auto_open=True)
    # fig_STFT_MH.write_html('STFT_MH.html', auto_open=True)



"""
This code reads in a folder of csv files, filters out the columns that are not in the list 'meas_sig', creates a folder for each file and then creates spectograms and STFT plots for each file. The spectograms and STFT plots are saved as html files in the respective folders. Finally, the folderpath is opened. 
 Step-by-step: 
1. Set parameters for the code: read_subfolders, Fs, f_resolution, t_resolution, max_plot_res, fmin, fmax, Time_min, Time_max, auto_open, meas_sig, and folderpath. 
2. If read_subfolders is False, get all the csv files in the folderpath. If read_subfolders is True, get all the csv files in the folderpath and its subfolders. 
3. For each file in the folderpath: 
    3a. Read in the csv file and filter out the columns that are not in the list 'meas_sig'. 
    3b. Create a folder for the file. 
    3c. Fix the time axis of the dataframe. 
    3d. Slice the dataframe to the specified time range. 
    3e. Create spectograms and STFT plots for the file. 
    3f. Save the spectograms and STFT plots as html files in the respective folder. 
4. Open the folderpath.
"""
def main():

    folderpath = r"V:\HW_Infrastructure\Analog_Team\ARC_Data\Results\Venus3\11.4kW SEDSP with FHB 74861BDA\RSS Arc Detection\Digital Records 04 (11-06-2023)"  # Folder path
    read_subfolders = True  # Boolean to determine if subfolders should be read
    Fs = 100e3 # Sampling frequency
    f_resolution = 100  # Frequency resolution
    t_resolution = 1e-3  # Time resolution
    max_plot_res = 1000  # Maximum plot resolution
    fmin = 0  # Minimum frequency
    fmax = 250e3  # Maximum frequency
    Time_min = 0  # Minimum time
    Time_max = 100  # Maximum time
    auto_open =False   # Boolean to determine if files should be opened automatically
    meas_sig = [ 'CH1', 'CH2']  # Measurement signals for scope
    add_spi_cols_and_time = True  # Boolean to determine if SPI columns should be added
    spi_cols=['CT Meas','CT Meas Filtered','Shunt Meas','Shunt Meas Filtered','Vdc Meas','Vdc Meas Filtered']
    SPI=True
    add_spi_cols=True #Boolean to determine if SPI columns should be added

    string_for_search='SPI' #filter files with this string
    apply_rolling_avg=True
    Plot_Zero_span=True
    zero_span_arr=[800,1000,1200,2000,4e3]

    Rolling_avg_pts_arr=[16,32,64,128,256]
     # Determine which files to read
    if read_subfolders == False:
        all_filespath = glob.glob(folderpath + '\*.csv')
    else:
        all_filespath = []
        for subdir, dirs, files in os.walk(folderpath):
            for file in files:
                full_path = os.path.join(subdir, file)
                if file.endswith('.csv'):
                    if string_for_search in file:
                        all_filespath.append(full_path)
     # Iterate through each file
    for file in all_filespath:
        df = pd.read_csv(file)
         # Drop columns that are not in the measurement signals


        if add_spi_cols:
            df=df.shift(1)
            df.iloc[0:1, ] = [float(x) for x in df.columns]
            df.columns= spi_cols
            meas_sig=spi_cols

        if SPI==True:
            df['Time']=df.index/Fs
            cols = df.columns.tolist()
            cols = cols[-1:] + cols[:-1]
            df = df[cols]
            #change the
        for col in df.columns:
            if col.lower() in 'time':
                continue
            if col not in meas_sig:
                df = df.drop(columns=col)
         # Get the file name
        file_name = os.path.basename(file).replace('.csv', '')
        print(F"File name is: {file_name}")
         # Create a folder for the file
        if not os.path.exists(folderpath+'\\'+file_name):
            os.mkdir(folderpath+'\\'+file_name)
        os.chdir(folderpath+'\\'+file_name)
         # Fix the time axis
        df = fix_time_axis(df, Fs)
         # Slice the dataframe by time
        df = df.loc[(df['Time'] >= Time_min) & (df['Time'] <= Time_max)]

        #plot all the data using plotly express
        spec_res = int(len(df) / max_plot_res)
        print(F"Spacing is: spec_res={spec_res})")
        if spec_res == 0:
            print(F"Spacing is 0 ploting full res")
            spec_res = 1
        fig = px.line(df[::spec_res], x="Time", y=meas_sig, title=f"Time domain plot of {file_name}")
        fig.write_html(f"0) Time domain {file_name} .html", auto_open=auto_open)
         # Generate spectograms
        fig_spectograms, trace_noam, Zraw, f, t = Noam_specogram(df, Fs, f_resolution, t_resolution, meas_sig, max_plot_res, fmin, fmax)
         # Slice the frequency
        Zraw, f = slice_freq(Zraw, f, fmin, fmax)
         # Calculate the magnitude of the raw data
        ZdBm = NDF_V4.Z_mag_calc(Zraw)
         # Write the spectograms to html
        for index, fig in enumerate(fig_spectograms):
            fig.write_html(f"1.{index}) Spectogram {meas_sig[index]} {file_name} .html", auto_open=auto_open)
         # Generate STFT figures
        fig_STFT_AVG, fig_STFT_MH = stft2(df, ZdBm, f, t, f_resolution, t_resolution, max_plot_res*10000,meas_sig)
         # Update the STFT figures
        config = dict({'scrollZoom': True})
        fig_STFT_MH.update_xaxes(title_text="Time [s]")
        fig_STFT_MH.update_yaxes(title_text="Frequency [Hz]")
        fig_STFT_MH.update_layout(title_text="STFT Max Hold")
        fig_STFT_MH.update_layout(title='All scope channels')
        fig_STFT_AVG.update_xaxes(title_text="Time [s]")
        fig_STFT_AVG.update_yaxes(title_text="Frequency [Hz]")
        fig_STFT_AVG.update_layout(title_text="STFT Avg Hold")
        fig_STFT_AVG.update_layout(title='All scope channels')
        fig_STFT_AVG['layout']['annotations'][0]['text'] = ''.join([str(x)+' ' for x in meas_sig])
        fig_STFT_MH['layout']['annotations'][0]['text'] = ''.join([str(x)+' ' for x in meas_sig])
         # Write the STFT figures to html
        fig_STFT_AVG.write_html(f"2) STFT_AVG_ALL_CH  {file_name} .html", auto_open=auto_open)
        fig_STFT_MH.write_html(f"3) STFT_MH_ALL_CH   {file_name} .html", auto_open=auto_open)

        if Plot_Zero_span:

            df_fft = NDF_V4.ZeroSpan_calc(ZdBm, meas_sig, t, f, zero_span_arr,meas_sig)

            fft_zero_span_fig, data_out_zero_span = NDF_V4.data_plot_streamlit(df_fft, 'Zero Span FFT', 't',
                                                                               max_plot_res)
            fft_zero_span_fig['layout'].update(title=' fft zero span')
            fft_zero_span_fig.write_html(f"4) Zero span {file_name} .html", auto_open=auto_open,config=config)

            if apply_rolling_avg:
                fig_zero_span_avg = go.Figure()
                for col in df_fft.columns:
                    if col == 't':
                        continue
                    else:
                        for window_to_avg in Rolling_avg_pts_arr:
                            fig_zero_span_avg.add_trace(
                                go.Scatter(
                                    visible=True,
                                    name=col + "avg win = " + str(window_to_avg),
                                    x=df_fft['t'].to_numpy(),
                                    y=df_fft[col].rolling(window_to_avg).mean().to_numpy()))

                fig_zero_span_avg['layout'].update(title=' fft zero span Avg')
                txt = f"5) Zero span Avg {file_name} .html"
                fig_zero_span_avg.write_html(txt, auto_open=auto_open,config=config)

     # Open the folder
    os.startfile(folderpath)

if __name__=='__main__':
    main()