# write spectogram to first coulm in a dataframe
import glob
import math
import os
import time
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from scipy import signal
from warnings import simplefilter
from progressbar import progressbar

import NDF_V4
from my_pyplot import omit_plot as _O, plot as _P, print_chrome as _PC, clear as _PP, print_lines as _PL, send_mail as _SM
import Plot_Graphs_with_Sliders as _G
import my_tools

EPSILON = 0.000000001
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
W = '\033[0m'  # white (normal)
R = '\033[31m'  # red
G = '\033[32m'  # green
O = '\033[33m'  # orange
B = '\033[34m'  # blue
P = '\033[35m'  # purpleW = '\033[0m'  # white (normal)
R = '\033[31m'  # red
G = '\033[32m'  # green
O = '\033[33m'  # orange
B = '\033[34m'  # blue
P = '\033[35m'  # purple


def sample_df(df, sample_interval, skip_interval):
    # Initialize variables
    current_time = df['Time'].iloc[0]
    sampled_df = pd.DataFrame(columns=df.columns)
    # Iterate through rows in the dataframe
    for index, row in df.iterrows():
        # Check if the current row's time is within the sample interval
        if row['Time'] >= current_time:
            # Append the row to the sampled dataframe
            sampled_df = sampled_df.append(row)
            # Update the current time by adding the sample interval
            current_time += sample_interval
        else:
            # Skip rows that fall within the skip interval
            continue
        # Update the current time by adding the skip interval
        current_time += skip_interval
    return sampled_df


def sample_df_with_main_loop(df, sample_interval, skip_interval, skip_interval_100):
    # Convert sample and skip intervals to seconds
    sample_interval_sec = sample_interval
    skip_interval_sec = skip_interval
    skip_interval_100_sec = skip_interval_100
    # Initialize variables
    current_time = df['Time'].iloc[0]
    skip_counter = 0
    sampled_df = pd.DataFrame(columns=df.columns)
    # Iterate through rows in the dataframe
    for index, row in df.iterrows():
        # Check if the current row's time is within the sample interval
        if row['Time'] >= current_time:
            # Append the row to the sampled dataframe
            sampled_df = sampled_df.append(row)
            # Update the current time by adding the sample interval
            current_time += sample_interval_sec
        else:
            # Skip rows that fall within the skip interval
            continue
        # Update skip counter
        skip_counter += 1
        # Check if 100 skips have occurred
        if skip_counter % 100 == 0:
            # Update the current time by adding the additional skip interval
            current_time += skip_interval_100_sec
        # Update the current time by adding the regular skip interval
        current_time += skip_interval_sec
    return sampled_df


class Goertzel:
    """Contains static methods for Goertzel calculations and represents a
    "classic" Goertzel filter.
    """

    @staticmethod
    def kernel(samples, koef, v_n1, v_n2):
        """The "kernel" of the Goertzel recursive calculation.  Processes
        `samples` array of samples to pass through the filter, using the
        `k` Goertzel coefficient and the previous (two) results -
        `v_n1` and `v_n2`.  Returns the two new results.
        """

        for samp in samples:
            v_n1, v_n2 = koef * v_n1 - v_n2 + samp, v_n1
        return v_n1, v_n2

    @staticmethod
    def VOLT(koef, v_n1, v_n2, nsamp):
        """Calculates (and returns) the 'dBm', or 'dBmW' - decibel-milliwatts,
        a power ratio in dB (decibels) of the (given) measured power
        referenced to one (1) milliwat (mW).
        This uses the audio/telephony usual 600 Ohm impedance.-> i used 50 not 600
        """
        amp_x = v_n1 ** 2 + v_n2 ** 2 - koef * v_n1 * v_n2
        if amp_x < EPSILON:
            amp_x = EPSILON
        # return 10 * np.log10(2 * amp_x * 1000 / (50*nsamp**2))

        return amp_x

    def dbm(koef, v_n1, v_n2, nsamp):
        """Calculates (and returns) the 'dBm', or 'dBmW' - decibel-milliwatts,
        a power ratio in dB (decibels) of the (given) measured power
        referenced to one (1) milliwat (mW).
        This uses the audio/telephony usual 600 Ohm impedance.-> i used 50 not 600
        """
        amp_x = v_n1 ** 2 + v_n2 ** 2 - koef * v_n1 * v_n2
        if amp_x < EPSILON:
            amp_x = EPSILON
        return 10 * np.log10(2 * amp_x * 1000 / (50 * nsamp ** 2))

        return amp_x

    @staticmethod
    def proc_samples_k(samples, koef):
        """Processe the given `samples` with the given `koef` Goertzel
        coefficient, returning the dBm of the signal (represented, in full,
        by the `samples`).
        """
        v_n1, v_n2 = Goertzel.kernel(samples, koef, 0, 0)
        return Goertzel.dbm(koef, v_n1, v_n2, len(samples))

    @staticmethod
    def calc_koef(freq, fsamp):
        """Calculates the Goertzel coefficient for the given frequency of the
        filter and the sampling frequency.
        """
        return 2 * math.cos(2 * math.pi * freq / fsamp)

    @staticmethod
    def process_samples(samples, freq, fsamp):
        """Processe the given +samples+ with the given Goertzel filter
        frequency `freq` and sample frequency `fsamp`, returning the
        dBm of the signal (represented, in full, by the `samples`).
        """
        return Goertzel.proc_samples_k(samples, Goertzel.calc_koef(freq, fsamp))

    def __init__(self, freq, fsamp):
        """To construct, give the frequency of the filter and the sampling
        frequency
        """
        if freq >= fsamp / 2:
            raise Exception("f is too big")
        self.freq, self.fsamp = freq, fsamp
        self.koef = Goertzel.calc_koef(freq, fsamp)
        self.zprev1 = self.zprev2 = 0

    def reset(self):
        """Reset for a new calculation"""
        self.zprev1 = self.zprev2 = 0

    def process(self, smp):
        """Process the given array of samples, return dBm"""
        self.zprev1, self.zprev2 = Goertzel.kernel(smp, self.koef, self.zprev1, self.zprev2)
        return Goertzel.dbm(self.koef, self.zprev1, self.zprev2, len(smp))


class GoertzelSampleBySample:
    """Helper class to do Goertzel algorithm sample by sample"""

    def __init__(self, freq, fsamp, nsamp, alpha):
        """I need Frequency, sampling frequency and the number of
        samples that we shall process"""
        self.freq, self.fsamp, self.nsamp = freq, fsamp, nsamp
        self.koef = 2 * math.cos(2 * math.pi * freq / fsamp)
        self.cnt_samples = 0
        self.zprev1 = self.zprev2 = 0
        self.Erg = 0
        self.ErgFiltered = 0
        self.alpha = alpha
        self.Enable_prints = False

    def process_sample(self, samp):
        """Do one sample. Returns dBm of the input if this is the final
        sample, or None otherwise."""
        Z = self.koef * self.zprev1 - self.zprev2 + (samp)
        self.zprev2 = self.zprev1
        self.zprev1 = Z
        self.cnt_samples += 1
        if self.cnt_samples == self.nsamp:
            self.cnt_samples = 0
            self.Erg = Goertzel.VOLT(self.koef, self.zprev1, self.zprev2, self.nsamp)
            self.ErgFiltered = self.alpha * self.Erg + (1 - self.alpha) * self.ErgFiltered
            return 1
        return None

    def reset(self):
        """Reset for a new calculation"""
        self.zprev1 = 0
        self.zprev2 = 0


def stft(dataframe, ZdBm, f, t, f_resolution, t_resolution, max_plot_res):
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
        res_name_arr = dataframe.loc[:, ~dataframe.columns.str.contains('Time')].columns
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


def stft2(dataframe, ZdBm, f, t, f_resolution, t_resolution, max_plot_res, meas_sig):
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
            i = np.where(channel == channels)[0][0]
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


def ZeroSpan_calc2(Z_arr, Z_name_arr, t, f, zero_span_arr, meas_sig, df):
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
            i = np.where(channel == channels)[0][0]

        for col_fft in Z_name_arr:
            Z_dBm = Z_arr[z_ind]
            z_ind = z_ind + 1
            if meas_sig in col_fft:
                for ff in f_arr:
                    ind = np.where(f == ff)[0][0]
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
                    ind = np.where(f == ff)[0][0]
                    # ind = int(ff / (f[1] - f[0]))
                    z_dBm = (Z_dBm[ind, :])
                    df_fft[col_fft + ' @ ' + str(round(float(f[ind] / 1000), 1)) + ' [kHz]'] = z_dBm
                # print(str(ind))

                if flag == 0:
                    flag = 1
                    df_fft['t'] = t

    end = time.time()

    return (df_fft)


def Noam_specogram(df, Fs, f_resolution, t_resolution, meas_sig, max_plot_res, fmin, fmax, Time_min=0, file=''):
    fft_win = int(Fs / f_resolution)
    fft_win_overlap = int(fft_win - (t_resolution * Fs))
    if fft_win > len(df):
        fft_win = int(len(df) / 10)
        fft_win_overlap = int(fft_win * 0.99)

    print('fft_win = ' + str(fft_win))
    print('fft_win_overlap = ' + str(fft_win_overlap))
    # get t resolution

    res_name_arr = []
    (t, f, Zraw, res_name_arr) = NDF_V4.df_stft_RAW_calc(df, Fs, fft_win, fft_win_overlap, df.loc[:, ~df.columns.str.contains('Time')].columns)
    if Time_min > 0:
        t = t + Time_min
    ZdBm = NDF_V4.Z_mag_calc(Zraw)
    Temp_fig_arr = []
    trace_arr = []

    for i in range(len(ZdBm)):
        Temp_fig, trace = NDF_V4.spectrogram_plot_for_streamlit(ZdBm[i], t, f, max_plot_res, fmax, fmin, meas_sig[i], file)
        trace_arr.append(trace)
        Temp_fig_arr.append(Temp_fig)

    return Temp_fig_arr, trace_arr, Zraw, f, t


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


def fix_time_axis(df, Fs):
    if 'Time'.upper() not in [x.upper() for x in df.columns]:
        df['Time'] = df.index * 1 / Fs
    else:
        # find which column is time
        for col in df.columns:
            if 'Time'.upper() in col.upper():
                df = df.rename({col: 'Time'}, axis=1)  # new method

                df.Time = df.Time - df.Time.min()

    return df


def slice_freq(Zraw, f, fmin, fmax):
    f_min_ind = int(fmin / (f[1] - f[0]))
    f_max_ind = int(fmax / (f[1] - f[0])) + 1
    f = f[f_min_ind:f_max_ind]
    for i in range(len(Zraw)):
        Zraw[i] = Zraw[i][f_min_ind:f_max_ind]

    return Zraw, f


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


def main(folderpath=None):
    if folderpath is None:
        folderpath = r'M:\Users\HW Infrastructure\PLC team\ARC\Temp-Eddy\AC Arcs\Automation\_TEMP'
    print('General params are:')
    read_subfolders = True  # Bug in code: no string filtering
    SCOPE_FS = 50e3
    SPI_FS = 100e3
    f_resolution = 100  # Frequency resolution
    t_resolution = 1e-3  # Time resolution
    max_plot_res_for_spi = 10000  # Maximum plot resolution
    max_plot_res_for_scope = 2000  # Maximum plot resolution
    fmin = 0  # Minimum frequency
    fmax = 50e3  # Maximum frequency
    Time_min = 0  # Minimum time
    Time_max = 100  # Maximum time
    auto_open = False  # Boolean to determine if files should be opened automatically
    N = 128
    alpha = 1
    pic_pixel = -1080
    goretzel_calc = False
    meas_sig_scope = ['CH1', 'CH2']  # Measurement signals for scope
    meas_sig_spi = ['Current', 'Voltage']  # Measurement signals for scope
    spi_cols = ['Current', 'Voltage']
    # meas_sig_change = ['Voltage', 'Current']  # Measurement signals for scope
    meas_sig_change = None
    SPI = True
    convert_adc = True
    add_spi_cols_and_time = True  # Boolean to determine if SPI columns should be added
    add_spi_cols = False  # Boolean to determine if SPI columns should be added
    # print all the params
    print('read_subfolders=', read_subfolders)
    print('SCOPE_FS=', SCOPE_FS)
    print('SPI_FS=', SPI_FS)
    print('f_resolution=', f_resolution)
    print('t_resolution=', t_resolution)
    print('max_plot_res_for_spi=', max_plot_res_for_spi)
    print('max_plot_res_for_scope=', max_plot_res_for_scope)
    print('fmin=', fmin)
    print('fmax=', fmax)
    print('Time_min=', Time_min)
    print('Time_max=', Time_max)
    print('auto_open=', auto_open)
    print('N=', N)
    print('alpha=', alpha)
    print('goretzel_calc=', goretzel_calc)
    print('meas_sig_scope=', meas_sig_scope)
    print('add_spi_cols_and_time=', add_spi_cols_and_time)
    print('meas_sig_spi=', meas_sig_spi)
    print('spi_cols=', spi_cols)
    print('SPI=', SPI)
    print('add_spi_cols=', add_spi_cols)

    if SPI:
        string_for_search = 'spi'  # filter files with this string
    else:
        string_for_search = 'scope'

    apply_rolling_avg = True
    Plot_Zero_span = True
    zero_span_arr = [10e3, 28e3, 36e3, 40e3, 44e3]
    Rolling_avg_pts_arr = [16, 32, 64, ]
    # Determine which files to read

    if read_subfolders == False:
        all_filespath = glob.glob(folderpath + '\*.csv')
    else:
        all_filespath = []
        for subdir, dirs, files in os.walk(folderpath):
            for file in files:
                full_path = os.path.join(subdir, file)
                if file.endswith('.txt') or file.endswith('.csv'):
                    if string_for_search in file.lower():
                        all_filespath.append(full_path)
    # Iterate through each file
    # print all files with \n in between
    # remove the
    for file in all_filespath:
        print(O + file + W)

    for file in all_filespath:
        is_spi = False
        meas_sig = []
        if 'terminal' in file.lower():
            continue
        if 'read me' in file.lower():
            continue
        df = pd.read_csv(file)
        # Drop columns that are not in the measurement signals

        if '.csv' in file and 'scope' in file.lower():
            file_name = os.path.basename(file).replace('.csv', '')
            is_scope = True
            Fs = SCOPE_FS
            meas_sig = meas_sig_scope
            max_plot_res = max_plot_res_for_scope
        else:
            if '.csv' in file:
                file_name = os.path.basename(file).replace('.csv', '')
            else:
                file_name = os.path.basename(file).replace('.txt', '')
            max_plot_res = max_plot_res_for_spi
            is_spi = True
            Fs = SPI_FS
            meas_sig = meas_sig_spi
        print(G + f"File name is: {file_name}" + W)

        if is_spi:
            if add_spi_cols:
                df = df.shift(1)
                df.iloc[0:1, ] = [float(x) for x in df.columns]
                df.columns = spi_cols
                meas_sig = spi_cols

            if SPI == True:
                df['Time'] = df.index / Fs
                if convert_adc and SPI:
                    df['Current'] = df['Current'] - 7767.772543
                    df['Current'] = df['Current'] / 385.78962239147813
                    df['Voltage'] = df['Voltage'] - 7780
                    df['Voltage'] = df['Voltage'] * 0.055
                cols = df.columns.tolist()
                cols = cols[-1:] + cols[:-1]
                df = df[cols]
        # change the

        for col in df.columns:
            if col.lower() in 'time':
                continue
            if col not in meas_sig:
                df = df.drop(columns=col)

        df = df.astype(float)
        # rename COL 'VARC' TO 'Vout'
        # try:
        #     df.rename(columns={'Varc':'Vout'}, inplace=True)
        #      meas_sig = ['Iarc', 'Vout', 'Lrx']  # Measurement signals for scope
        # except:
        #         meas_sig = ['Iarc', 'Vout', 'Lrx']
        # Create a folder for the file
        if not os.path.exists(folderpath + '\\' + file_name):
            os.mkdir(folderpath + '\\' + file_name)
        os.chdir(folderpath + '\\' + file_name)
        # Fix the time axis
        df = fix_time_axis(df, Fs)
        # Slice the dataframe by time
        df = df.loc[(df['Time'] >= Time_min) & (df['Time'] <= Time_max)]

        # plot all the data using plotly express
        spec_res = int(len(df) / max_plot_res)

        print(F"Spacing is: spec_res={spec_res})")
        if spec_res == 0:
            print(F"Spacing is 0 ploting full res")
            spec_res = 1
        spec_res_temp = 50
        if meas_sig_change is not None:
            if SPI:
                df = df.rename(columns={meas_sig_spi[i]: meas_sig_change[i] for i in range(len(meas_sig_spi))})
            else:
                df = df.rename(columns={meas_sig_scope[i]: meas_sig_change[i] for i in range(len(meas_sig_scope))})
            meas_sig = meas_sig_change
        fig = px.line(df[::spec_res_temp], x="Time", y=meas_sig, title=f"Time domain plot of {file_name}")
        fig.write_html(f"0) Time domain {file_name} .html", auto_open=auto_open)
        # Generate spectograms
        fig_spectograms, trace_noam, Zraw, f, t = Noam_specogram(df, Fs, f_resolution, t_resolution, meas_sig, max_plot_res, fmin, fmax, Time_min, file.split('\\')[-1])
        # Slice the frequency
        Zraw, f = slice_freq(Zraw, f, fmin, fmax)
        # Calculate the magnitude of the raw data
        ZdBm = NDF_V4.Z_mag_calc(Zraw)
        # Write the spectograms to html
        for index, fig in enumerate(fig_spectograms):
            datax = fig.data[0]['x']
            datay = fig.data[0]['y']
            dataz = fig.data[0]['z']
            # slice the daya y

            # binsize=200e3/128
            # start_frame=40e3-(binsize/2)
            # stop_frame=40e3+(binsize/2)
            # fig.add_selection(x0=min(datax), y0=start_frame, x1=max(datax), y1=stop_frame,
            #                   line=dict(
            #                       color="black",
            #                       width=2,
            #                       dash="dash",
            #                   ))
            fig.write_html(f"1.{index}) Spectogram {meas_sig[index]} {file_name} .html", auto_open=auto_open)
            if pic_pixel > 0:
                fig.write_image(f"1.{index}) Spectogram photo {meas_sig[index]} {file_name} .svg", width=int(1.777778 * pic_pixel), height=pic_pixel)

        # Generate STFT figures
        fig_STFT_AVG, fig_STFT_MH = stft2(df, ZdBm, f, t, f_resolution, t_resolution, max_plot_res * 5000, meas_sig)
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
        fig_STFT_AVG['layout']['annotations'][0]['text'] = ''.join([str(x) + ' ' for x in meas_sig])
        fig_STFT_MH['layout']['annotations'][0]['text'] = ''.join([str(x) + ' ' for x in meas_sig])
        # Write the STFT figures to html
        fig_STFT_AVG.write_html(f"2) STFT_AVG_ALL_CH  {file_name} .html", auto_open=auto_open)
        fig_STFT_MH.write_html(f"3) STFT_MH_ALL_CH   {file_name} .html", auto_open=auto_open)

        if Plot_Zero_span:

            df_fft = NDF_V4.ZeroSpan_calc(ZdBm, meas_sig, t, f, zero_span_arr, meas_sig)

            fft_zero_span_fig, data_out_zero_span = NDF_V4.data_plot_streamlit(df_fft, 'Zero Span FFT', 't',
                                                                               max_plot_res)
            fft_zero_span_fig['layout'].update(title=' fft zero span')
            fft_zero_span_fig.write_html(f"4) Zero span {file_name} .html", auto_open=auto_open, config=config)

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
                fig_zero_span_avg.write_html(txt, auto_open=auto_open, config=config)

        # Open the folder

        config = {'scrollZoom': True}
        Freqs_for_goertzel_arr = zero_span_arr

        Fixed_point_goertzel = False
        # goertzel_fig = NDF_V4.goertzel_calc(df, Freqs_for_goertzel_arr,
        #                                     df.loc[:, ~df.columns.str.contains('Time')].columns,
        #                                     Fixed_point_goertzel, Fs, N, alpha)

        if goretzel_calc:
            counter_calc = 0
            # from window to f resulotion

            for signal in meas_sig:
                counter_calc += 1
                fig = NDF_V4.make_fig(row=2, col=1, subplot_titles=['Goertzel', 'Goertzel with skip'], shared_xaxes=True)
                downsample_factor = int(Fs / Fs)  # int(Fs/200e3)

                for detect_freq in Freqs_for_goertzel_arr:

                    # FILTER THE DF BEFORE DOWN SAMPLING

                    df_cpy = df.copy()[::downsample_factor]
                    RX_array = df_cpy[signal].to_numpy()
                    Time_array = df_cpy['Time'].to_numpy()

                    Goertzel_fs = int(Fs / downsample_factor)
                    print(signal)
                    print('The Goertzel sampling frequency is:')
                    print(Goertzel_fs)
                    print('The Goertzel downsample factor is:')
                    print(downsample_factor)
                    f_resolution = Goertzel_fs / N
                    print('The Goertzel frequency resolution is:')
                    print(f_resolution)
                    fft_win_overlap = 0
                    t_resolution = (fft_win_overlap + N) / Goertzel_fs
                    print('The Goertzel time resolution is:')
                    print(str(t_resolution))

                    samp = NDF_V4.GoertzelSampleBySample(detect_freq, Goertzel_fs, N, alpha=alpha, window_bool=False)

                    dft_out_goertzel_1_TONE = []
                    Time_axis_1_TONE = []
                    for idx, sample in enumerate(RX_array):
                        temp = samp.process_sample(sample)
                        if temp is not None:
                            dft_out_goertzel_1_TONE.append(samp.ErgFiltered)
                            Time_axis_1_TONE.append(Time_array[idx])
                            samp.reset()
                    str(int(detect_freq / 1000)) + '[kHZ]'
                    dft_out_goertzel_with_iir_db = 10 * np.log10((np.array(dft_out_goertzel_1_TONE)))

                    fig.add_trace(go.Scattergl(x=Time_axis_1_TONE[:],
                                               y=dft_out_goertzel_with_iir_db[:],
                                               name='Gorezel ' + str(detect_freq) + ' FS=' + str(int(Goertzel_fs / 1000)) + ' [kHZ]' + " detection at" + str(
                                                   int(detect_freq / 1000)) + '[kHZ]', mode="lines",
                                               visible=True,
                                               showlegend=True), row=1, col=1)

                    if apply_rolling_avg:
                        dff = pd.DataFrame(dft_out_goertzel_with_iir_db, columns=[signal])

                        for window_to_avg in Rolling_avg_pts_arr:
                            fig.add_trace(
                                go.Scatter(
                                    visible=True,
                                    name=str(detect_freq) + " Rolling avg win = " + str(window_to_avg),
                                    x=Time_axis_1_TONE[:],
                                    y=dff[signal].rolling(window_to_avg).mean().to_numpy()), row=1, col=1)

                    # Goertzel with skip
                    df_cpy = df.copy()
                    # df_resampled= sample_df_with_main_loop(df_cpy[::downsample_factor],852e-6,1e-6,2e-3)
                    # df_resampled = sample_df(df_cpy[::downsample_factor], 852e-6, 1e-6)
                    df_resampled = df_cpy[::downsample_factor]
                    print('resampled df')
                    RX_array = df_resampled[signal].to_numpy()
                    Time_array = df_resampled['Time'].to_numpy()

                    samp = NDF_V4.GoertzelSampleBySample(detect_freq, Goertzel_fs, N, alpha=alpha, window_bool=False)

                    dft_out_goertzel_1_TONE = []
                    Time_axis_1_TONE = []
                    Windows_skip = 852e-6
                    Numberofpointto_skip = 170  # Goertzel_fs*Windows_skip
                    counter = 0
                    windows_skip = 652e-6
                    dftonPts = 200  # Goertzel_fs * Windows_skip
                    for idx, sample in enumerate(RX_array):
                        if (counter >= 0 and counter <= dftonPts):
                            temp = samp.process_sample(sample)
                            if temp is not None:
                                dft_out_goertzel_1_TONE.append(samp.ErgFiltered)
                                Time_axis_1_TONE.append(Time_array[idx])
                                samp.reset()
                            if counter == dftonPts:
                                counter = -Numberofpointto_skip
                        else:
                            counter += 1
                            continue
                        counter += 1

                    str(int(detect_freq / 1000)) + '[kHZ]'
                    dft_out_goertzel_with_iir_db = 10 * np.log10((np.array(dft_out_goertzel_1_TONE)))

                    fig.add_trace(go.Scattergl(x=Time_axis_1_TONE[:],
                                               y=dft_out_goertzel_with_iir_db[:],
                                               name='Gorezel skip ' + str(detect_freq) + ' FS=' + str(
                                                   int(Goertzel_fs / 1000)) + ' [kHZ]' + " detection at" + str(
                                                   int(detect_freq / 1000)) + '[kHZ]', mode="lines",
                                               visible=True,
                                               showlegend=True), row=2, col=1)

                    if apply_rolling_avg:
                        dff = pd.DataFrame(dft_out_goertzel_with_iir_db, columns=[signal])
                        for window_to_avg in Rolling_avg_pts_arr:
                            fig.add_trace(
                                go.Scatter(
                                    visible=True,
                                    name=str(detect_freq) + " Rolling avg win = " + str(window_to_avg),
                                    x=Time_axis_1_TONE[:],
                                    y=dff[signal].rolling(window_to_avg).mean().to_numpy()), row=2, col=1)

                    samp = NDF_V4.GoertzelSampleBySample(detect_freq, Goertzel_fs, N, alpha=alpha, window_bool=True)

                    dft_out_goertzel_1_TONE = []
                    Time_axis_1_TONE = []
                    Windows_skip = 852e-6
                    Numberofpointto_skip = 170  # Goertzel_fs*Windows_skip
                    counter = 0
                    windows_skip = 652e-6
                    dftonPts = 200  # Goertzel_fs * Windows_skip
                    for idx, sample in enumerate(RX_array):
                        if (counter >= 0 and counter <= dftonPts):
                            temp = samp.process_sample(sample)
                            if temp is not None:
                                dft_out_goertzel_1_TONE.append(samp.ErgFiltered)
                                Time_axis_1_TONE.append(Time_array[idx])
                                samp.reset()
                            if counter == dftonPts:
                                counter = -Numberofpointto_skip
                        else:
                            counter += 1
                            continue
                        counter += 1

                    str(int(detect_freq / 1000)) + '[kHZ]'
                    dft_out_goertzel_with_iir_db = 10 * np.log10((np.array(dft_out_goertzel_1_TONE)))

                    fig.add_trace(go.Scattergl(x=Time_axis_1_TONE[:],
                                               y=dft_out_goertzel_with_iir_db[:],
                                               name='Gorezel skip +window ' + str(detect_freq) + ' FS=' + str(
                                                   int(Goertzel_fs / 1000)) + ' [kHZ]' + " detection at" + str(
                                                   int(detect_freq / 1000)) + '[kHZ]', mode="lines",
                                               visible=True,
                                               showlegend=True), row=2, col=1)

                    if apply_rolling_avg:
                        dff = pd.DataFrame(dft_out_goertzel_with_iir_db, columns=[signal])
                        for window_to_avg in Rolling_avg_pts_arr:
                            fig.add_trace(
                                go.Scatter(
                                    visible=True,
                                    name=str(detect_freq) + " +window Rolling avg win = " + str(window_to_avg),
                                    x=Time_axis_1_TONE[:],
                                    y=dff[signal].rolling(window_to_avg).mean().to_numpy()), row=2, col=1)

                txt1 = "N=" + str(N) + " alpha=" + str(alpha) + ' Goertzel'
                fig['layout'].update(title=txt1)
                txt = f"6.{counter_calc}) Goertzel   {file_name}  alpha={alpha}.html"
                fig.write_html(txt, auto_open=auto_open, config=config)
        print(G + F"Done: {file_name}" + W)


if __name__ == '__main__':
    # main()

    all_folders = []
    main_folder = r'M:\Users\HW Infrastructure\PLC team\ARC\Temp-Eddy\AC Arcs\Automation'
    all_folders.append(main_folder + '\\' + 'Arcs 01 - Jup2 (19-07-2023)')
    all_folders.append(main_folder + '\\' + 'Arcs 02 - Jup2 (23-07-2023)')
    all_folders.append(main_folder + '\\' + 'Arcs 03 - JPI (24-07-2023)')
    bar = progressbar.ProgressBar()
    for folder in bar(all_folders):
        print(B+'Starting new folder \n --------------------------------------------------------------------------------------------------------------------\n'+W)
        try:
            main(folderpath=folder)
        except:
            print(R+"Failed to run folder : "+folder+W)
            continue
