import os
import time
import pandas as pd
import numpy as np
import plotly
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import gc
import NDF_V2
import arc_th_calc
import log_file
from my_pyplot import omit_plot as _O, plot as _P, print_chrome as _PC, clear as _PP, print_lines as _PL, send_mail as _SM
import Plot_Graphs_with_Sliders as _G
import my_tools


read_all_folder = True
KILL_CHORME = False
auto_open_chrome_output = False
plot_on = False
scope_fs = 50e3
f_resolution = 100  # Spectrum resolution RBW
t_resolution = 0.001  # Time resolution
lab_folder = r"M:\Users\HW Infrastructure\PLC team\ARC\Temp-Eddy\New_Arc_Detection_Frequency 24_10_21 21_31"
lab_folder = r'M:\Users\LinoyL\Vdc_Criteria_new\MOHAMD_ARCS\RAW_DATA\S1\A1\S1_InvCb_7A'
meas_sig = 'RX Arc OUT'
plot_energy_rise = False
zero_span_arr = np.arange(100, 25e3, 100)
alpha_filter = 0.2875
go_to_file = 87

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

tstart = time.time()

path = lab_folder + '\\'
if not os.path.exists(lab_folder):
    os.makedirs(lab_folder)
test_name = 'files lab setup SPI'  # result will be here+filename
file_name_arr = []

if read_all_folder:
    file_arr_temp = []
    file_name_arr += [f for f in os.listdir(path) if (f.endswith('.csv') or f.endswith('.txt')) and 'terminal' not in f]
    for filename in file_name_arr:
        file_arr_temp += [os.path.splitext(filename)[0]]
    file_name_arr = []
    file_name_arr = file_arr_temp

tag_arr = file_name_arr
spi_ch_arr = ["Vdc"]
scope_ch_arr = [ "CH4"]
skip_scope = False
skip_spi = True


## <codecell> Read Data Frame from SCOPE or SPI Data CSV file
for file_index, filename in enumerate(file_name_arr):
    if file_index < go_to_file - 1:
        continue
   # results_path = path + test_name + '\\' + filename
    results_path = path  + '\\' + filename
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    if "SCOPE" in filename or "scope" in filename:
        if skip_scope==False:
            ch_arr = scope_ch_arr
            df = NDF_V2.scope_CSV_to_df(path, filename, scope_ch_arr, scope_ch_arr,True, scope_fs)  ##false mean that ypu dont have time col in your data
        else:
            continue
    elif "SPI" in filename or "spi" in filename:
        if skip_spi == False:
            spi_Fs = 50e3
            spi_filename = filename
            ch_arr = spi_ch_arr
            df = NDF_V2.spi_TXT_to_df(path, spi_filename,spi_ch_arr, spi_ch_arr, spi_Fs,plot_raw=True)
        else:
            continue
    else:
        continue

    ## <codecell> DataFrame Chunk
    t_start =0
    t_end = 100
    df1 = NDF_V2.df_Chunk(df, t_start, t_end)
    df = df1
    gc.collect()

    # var_win_factor = 100
    # avg_win_factor = 20

    var_win_factor = 5
    avg_win_factor = 5
    ## <codecell> DataFrame Simple Down Sample
    down_sample = 1
    df1 = df1.iloc[::down_sample]
    Fs = round(1 / (df1['Time'].iloc[1] - df1['Time'].iloc[0]))  # Data samples frequency
    ## <codecell> Reset Chunk Time
    df1 = NDF_V2.df_time_reset(df1, 'Time')
    tmax = max(df1['Time'])
    ## <codecell> Variance and std Calculations on RAW data+plot on time domian
    var_win_t = 0.05 #in sec
    win_size = int(var_win_t * Fs)
    df_var = NDF_V2.df_var_calc(df1, win_size, 'Time')
    df_var_dB = NDF_V2.df_dB_calc(df_var, 'Time')

    # %% Calculate of Variance and std averaging
    avg_win_t = 0.01
    win_size = int(avg_win_t * Fs)
    df_var_mean = NDF_V2.df_mean_calc(df_var, win_size, 'Time')
    df_var_mean_dB = NDF_V2.df_dB_calc(df_var_mean, 'Time')
    new_col = df_var_mean_dB.columns[1:]
    df_var_dB[new_col] = df_var_mean_dB[new_col]

    del df_var
    del df_var_mean
    del df_var_mean_dB
    gc.collect()
    # %% Plots - SCOPE RAW, Variance, Variance mean, Variance dB
    max_plot_res = 100000
    for channel in ch_arr:
        file_on = False

        scope_plot_data = NDF_V2.data_plot(df1, 'RAW Data', 'Time', max_plot_res, channel, plot_on, file_on, results_path)
        var_plot_data = NDF_V2.data_plot(df_var_dB, 'Variance', 'Time', max_plot_res, channel, plot_on, file_on,
                                      results_path)

        data_list = [scope_plot_data, var_plot_data]
        name_list = ['Scope RAW', 'Scope Variance [dB]']
        x_sync = True
        data_name = ' 01_Time Domain RAW Data Stats [RAW-Variance-Mean]'
        # NDF_V2.data_pane_plot(data_name, data_list, name_list, plot_on, x_sync, tag_arr, channel, results_path)

    del df_var_dB
    gc.collect()
    NDF_V2.kill_chrome(KILL_CHORME)
    ## <codecell> STFT Spectrogram Calculation of DataFrame



    fft_win = int(Fs / f_resolution)
    fft_win_overlap = int(fft_win - (t_resolution * Fs))
    # fft_win = 10000
    # fft_win_overlap = 5000

    if fft_win > len(df1):
        fft_win = int(len(df1) / 10)
        fft_win_overlap = int(fft_win * 0.99)
    res_name_arr = []
    (t, f, Zraw, res_name_arr) = NDF_V2.df_stft_RAW_calc(df1, Fs, fft_win, fft_win_overlap, ch_arr)
    f = f[f < 200000]

    ## <codecell> Z magnitude calculations

    calc_mag = True
    calc_phase = False
    if calc_mag:
        ZdBm = NDF_V2.Z_mag_calc(Zraw)

    # <codecell> Z phase calculations
    if calc_phase:
        phase_unwrap = False
        Zphase = NDF_V2.Z_phase_calc(Zraw, phase_unwrap)

    ## <codecell> Plot - Spectrogram
    spectrogram_on = True
    if spectrogram_on:
        fmin = 1
        fmax = 100000
        t_start = 0
        t_stop=5
        max_plot_res = 2500
        for i in range(len(Zraw)):
            # meas_sig = res_name_arr[i]
            if calc_mag:
                NDF_V2.scpectrogram_plot(ZdBm[i], t, f, max_plot_res, fmax, fmin, t_start,t_stop, plot_on, results_path)
            if calc_phase:
                NDF_V2.scpectrogram_plot(Zphase[i], t, f, max_plot_res, fmax, fmin, t_start, t_stop, plot_on,
                                         results_path, "01")
        NDF_V2.kill_chrome(KILL_CHORME)
    ## <codecell> ZeroSpan results for STFT results
    save_zero_span = True
    if calc_mag:
        df_fft = NDF_V2.ZeroSpan_calc(ZdBm, res_name_arr, t, f, zero_span_arr, ch_arr)
        if plot_energy_rise:
            energy_rise_keys = []
            energy_rise_values = []
            for key, value in df_fft.items():
                if key == 't':
                    continue
                elif '@' in key:
                    value = NDF_V2.Avg_no_overlap(value, 4)
                    energy_rise_keys.append(float(key[6:].split('[')[0]) * 1000)
                    energy_rise_values.append(max(arc_th_calc.plot_all(log_file.alpha_beta_filter(value, alpha=alpha_filter), 20, 15, 12)))
                else:
                    raise ValueError('A very specific bad thing happened.')
            fig = make_subplots(rows=1, cols=1, shared_xaxes=False)
            fig_name = '05 Energy Rise with min and max average FFTs'
            fig.add_trace(go.Scatter(y=energy_rise_values, x=energy_rise_keys, name='Energy Rise'), col=1, row=1)
            fig.update_layout(title=fig_name, title_font_color="#407294", title_font_size=40, legend_title="Plots:", legend_title_font_color="green")
            plotly.offline.plot(fig, config={'scrollZoom': True, 'editable': True}, filename=f'{results_path}\\{fig_name}.html', auto_open=auto_open_chrome_output)
            # zs_amp_str = str(zero_span_arr).replace(",", " -")
            # NDF_V2.Save_df_fft_mag(results_path, df_fft, filename)

    if calc_phase:
        #zero_span_phase_arr = [3e3, 4e3, 5.2e3, 15e3, 16.6e3, 20e3, 40e3,57e3,59e3,72e3,74e3,80e3, 106e3]
        zero_span_phase_arr = [4.6e3]
        df_fft_phase = NDF_V2.ZeroSpan_calc(Zphase, res_name_arr, t, f, zero_span_phase_arr, ch_arr)
        cols = df_fft_phase.columns[1:]
        df_fft_phase[cols] = df_fft_phase[cols].diff()
        zs_phase_str = str(zero_span_phase_arr).replace(",", " -")
        # NDF_V2.Save_df_fft_phase(results_path, df_fft_phase, filename)

    ## <codecell> Plot - Scope RAW, Zero Span FFT, Variance of Zero Span FFT, Mean of Zero Span FFT
    max_plot_res = 100000
    calc_mag = True
    if calc_mag:
        # for channel in ch_arr:
        var_win_t = 0.05
        win_size = int(var_win_t / t_resolution)
        fft_var = NDF_V2.df_var_calc(df_fft, win_size, 't')

        win_size = int(avg_win_t / t_resolution)
        for channel in ch_arr:
            file_on = False
            scope_plot_data = NDF_V2.data_plot(df1, 'SPI RAW Data', 'Time', max_plot_res, channel, plot_on, file_on,
                                            results_path)
            fft_plot_data = NDF_V2.data_plot(df_fft, 'Zero Span FFT', 't', max_plot_res, channel, plot_on, file_on,
                                          results_path)
            fft_var_plot_data = NDF_V2.data_plot(fft_var, 'Zero Span FFT Variance', 't', max_plot_res, channel, plot_on,
                                              file_on, results_path)

            data_list = [scope_plot_data, fft_plot_data]
            name_list = ['RAW Data', 'ZeroSpan FFT']
            data_list = [scope_plot_data, fft_plot_data, fft_var_plot_data]
            name_list = ['RAW Data', 'ZeroSpan FFT', 'ZeroSpan FFT Variance']
            x_sync = False
            # NDF_V2.data_pane_plot('03_mag - FFT Magnitude Zero Span Plots' + ' Hz freqs', data_list,
            #                    name_list, plot_on, x_sync, tag_arr, channel, results_path)

    max_plot_res = 100000
    if calc_phase:
        # for channel in ch_arr:
        var_win_t = 0.05
        win_size = int(var_win_t / t_resolution)
        fft_var = NDF_V2.df_var_calc(df_fft_phase, win_size, 't')

        avg_win_t = 0.1
        win_size = int(avg_win_t / t_resolution)
        fft_var_mean = NDF_V2.df_mean_calc(fft_var, win_size, 't')
        new_col = fft_var_mean.columns[1:]
        fft_var[new_col] = fft_var_mean[new_col]

        fft_mean = NDF_V2.df_mean_calc(df_fft_phase, win_size, 't')
        new_col = fft_mean.columns[1:]
        df_fft_phase[new_col] = fft_mean[new_col]

        del fft_var_mean
        del fft_mean
        gc.collect()
        for channel in ch_arr:
            file_on = False
            scope_plot_data = NDF_V2.data_plot(df1, 'Scope RAW Data', 'Time', max_plot_res, channel, plot_on, file_on,
                                            results_path)
            fft_phase_plot_data = NDF_V2.data_plot(df_fft_phase, 'Zero Span FFT', 't', max_plot_res, channel, plot_on,
                                                file_on, results_path)
            fft_var_plot_data = NDF_V2.data_plot(fft_var, 'Zero Span FFT Variance', 't', max_plot_res, channel, plot_on,
                                              file_on, results_path)

            data_list = [scope_plot_data, fft_phase_plot_data, fft_var_plot_data]
            name_list = ['RAW Data', 'ZeroSpan Phase', 'ZeroSpan Phase Variance']
            x_sync = False
            NDF_V2.data_pane_plot('03 FFT Phase Zero Span Plots for ' + zs_phase_str + ' Hz freqs', data_list,
                               name_list, plot_on, x_sync, tag_arr, channel, results_path)
        NDF_V2.kill_chrome(KILL_CHORME)

    ## <codecell> Sliding Window Magnitude Spectrum Analysis and Plots
    if calc_mag:
        MH_time = 0.1
        Overlap_time = 0.05

        win_size = int(MH_time / t_resolution)
        win_overlap = int(Overlap_time / t_resolution)
        max_plot_res = 100000

        for channel in ch_arr:
            df_MH_1 = pd.DataFrame(columns=['f'])
            df_AVG_1 = pd.DataFrame(columns=['f'])

            df_MH_1['f'] = f
            df_AVG_1['f'] = f

            indices = [i for i, elem in enumerate(res_name_arr) if channel in elem]
            if not indices:
                continue
            for i in indices:
                meas_sig = res_name_arr[i]
                if channel in meas_sig:
                    (df_MH_temp, df_AVG_temp, t_list, df_MIN_temp) = NDF_V2.sliding_spectrum(ZdBm[i], t, f, win_size, win_overlap,
                                                                             meas_sig)

                df_MH_1[df_MH_temp.columns] = df_MH_temp
                df_AVG_1[df_AVG_temp.columns] = df_AVG_temp

            del df_MH_temp
            del df_AVG_temp
            file_on = False
            scope_plot_data = NDF_V2.data_plot(df1, 'Scope RAW Data', 'Time', max_plot_res, channel, plot_on, file_on,
                                            results_path)
            MH1_plot_data = NDF_V2.data_plot(df_MH_1, 'Sliding Spectrum MH', 'f', max_plot_res, channel, plot_on, file_on,
                                          results_path)
            AVG1_plot_data = NDF_V2.data_plot(df_AVG_1, 'Sliding Spectrum AVG', 'f', max_plot_res, channel, plot_on,
                                           file_on, results_path)

            data_list = [scope_plot_data, MH1_plot_data]
            name_list = ['ZeroSpan Results', 'Sliding MH Spectrum dBm']
            # data_list=[ MH1_plot_data]
            # name_list=['Sliding MH Spectrum dBm']
            x_sync = False
            data_name = '02 Sliding FFT - MH Spectrum with ZeroSpan'
            NDF_V2.data_pane_plot(data_name, data_list, name_list, plot_on, x_sync, tag_arr, channel, results_path)

            data_list = [scope_plot_data, AVG1_plot_data]
            name_list = ['ZeroSpan Results', 'Sliding AVG Spectrum dBm']
            # data_list=[ AVG1_plot_data]
            # name_list=['Sliding AVG Spectrum dBm']
            x_sync = False
            data_name = '03 Sliding FFT - AVG Spectrum with ZeroSpan'
            NDF_V2.data_pane_plot(data_name, data_list, name_list, plot_on, x_sync, tag_arr, channel, results_path)

            data_list = [MH1_plot_data, AVG1_plot_data]
            name_list = ['Sliding MH Spectrum dBm', 'Sliding AVG Spectrum dBm']
            x_sync = False
            data_name = '04 Sliding FFT - MH and AVG Spectrum'
            NDF_V2.data_pane_plot(data_name, data_list, name_list, plot_on, x_sync, tag_arr, channel, results_path)
        NDF_V2.kill_chrome(KILL_CHORME)
    ## <codecell> Sliding Window Phase Spectrum Analysis and Plots
    slide_FFT_phase = True;
    if calc_phase and slide_FFT_phase:
        MH_time = 0.1
        Overlap_time = 0.05

        win_size = int(MH_time / t_resolution)
        win_overlap = int(Overlap_time / t_resolution)
        max_plot_res = 100000

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

            file_on = False
            scope_plot_data = NDF_V2.data_plot(df1, 'Scope RAW Data', 'Time', max_plot_res, channel, plot_on, file_on,
                                            results_path)
            MH1_plot_data = NDF_V2.data_plot(df_MH_1, 'Sliding Spectrum MH', 'f', max_plot_res, channel, plot_on, file_on,
                                          results_path)
            AVG1_plot_data = NDF_V2.data_plot(df_AVG_1, 'Sliding Spectrum AVG', 'f', max_plot_res, channel, plot_on,
                                           file_on, results_path)

            data_list = [scope_plot_data, MH1_plot_data]
            name_list = ['Scope RAW', 'Sliding MH Spectrum dBm']
            x_sync = False
            data_name = '03 Sliding FFT - MH Spectrum with Scope RAW'
            NDF_V2.data_pane_plot(data_name, data_list, name_list, plot_on, x_sync, tag_arr, channel, results_path)

            data_list = [scope_plot_data, AVG1_plot_data]
            name_list = ['Scope RAW', 'Sliding AVG Spectrum dBm']
            x_sync = False
            data_name = '03 Sliding FFT - AVG Spectrum with Scope RAW'
            NDF_V2.data_pane_plot(data_name, data_list, name_list, plot_on, x_sync, tag_arr, channel, results_path)

            data_list = [MH1_plot_data, AVG1_plot_data]
            name_list = ['Sliding MH Spectrum dBm', 'Sliding AVG Spectrum dBm']
            x_sync = True
            data_name = '04 Sliding FFT - MH and AVG Spectrum'
            NDF_V2.data_pane_plot(data_name, data_list, name_list, plot_on, x_sync, tag_arr, channel, results_path)

    NDF_V2.kill_chrome(KILL_CHORME)
    ## <codecell> End Sequence
    # f_resolution = 100  # Spectrum resolution RBW
    # t_resolution = 0.05  # Time resolution
    #
    # fft_win = int(Fs / f_resolution)
    # fft_win_overlap = int(fft_win - (t_resolution * Fs))
    # plot_on = True
    # waveletname = 'db9'
    # Maximum_decomposition_level = pywt.dwt_max_level(len(df), waveletname)
    # print("Maximum_decomposition_level is %s " % (Maximum_decomposition_level))
    # for col in df1.iloc[:, 1:]:
    #     df_wavelet = NDF_V2.Sliding_WDT(df1, 'Time', col, fft_win, fft_win_overlap, waveletname, 6)
    #     res = []
    #
    #     res = []
    #     for wavelet_col in df_wavelet.columns[2:]:
    #         res.append(
    #             go.Bar(
    #                 x=df_wavelet['Time'],
    #                 y=df_wavelet[col],
    #                 name=wavelet_col
    #             )
    #         )
    #     layout = go.Layout(
    #         barmode='group'
    #     )
    #     fig = go.Figure(data=res, layout=layout)
    #     config = {'scrollZoom': True, 'editable': True}
    #     # fig['layout'].update(title=string+ 'SNR' )
    #     fig['layout'].update(xaxis=dict(title='time'))
    #     fig['layout'].update(yaxis=dict(title='coeff power RMS'))
    #     # data_out.append(data)
    #     results_path = results_path + '/'
    #     if not os.path.exists(results_path):
    #         os.makedirs(results_path)
    #     print("Generated plot file")
    #     py.offline.plot(fig, auto_open=plot_on, config=config, filename=results_path + '/' + col + ' df_wavelet.html')
    #
    #     print('\nStop Run')
    #     tend = time.time()
    #     print('---| Runtime = ' + str(tend - tstart) + ' Sec')
        # df_wavelet2=igf2.DF_to_WDT_DF(df,'Time','Rec034_SPI_Vin',waveletname,5)
    ## <codecell> End Sequence
    for element in dir():
        if element[0:2] != "__":
            if element in ['spi_ch_arr', 'fft_plot_data', 'data_list', 'MH1_plot_data', 'Zraw', 'AVG1_plot_data', 'ZdBm', 'data_list', 'scope_plot_data', 'df_fft',
                           'df', 'df1', 'down_sample']:
                del globals()[element]
            else:
                print(f'element =  {element}')
### Don't delete: 'zero_span_arr', 'data_list', 'scope_plot_data', 'scope_ch_arr'
    gc.collect()
