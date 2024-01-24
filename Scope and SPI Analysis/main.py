import os
import gc
import sys
from datetime import datetime
from statistics import mean
import plotly
import threading
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.ndimage.filters import uniform_filter1d
import scipy
import NDF_V2
import log_file
from my_pyplot import omit_plot as _O, plot as _P, print_chrome as _PC, clear as _PP, print_lines as _PL, send_mail as _SM
import Plot_Graphs_with_Sliders as _G
import my_tools


filter_sex = ['Selected arcs for scope FFT', 'Noise floor No Telems', 'Force Telemetries discrete', 'False Alarms case 9']
filter_mana = [False, 140, 'TX2-RX3', filter_sex[0], 2200]
use_plots_no_threads = True
avg_no_energy_rise = True
arcs = True
frag_factor = 1
frag_n_sample = 250
output_text = False
auto_open_chrome_output = True
print_prints = use_plots_no_threads and True
spectrogram_EnergyRise_BOTH = 1   # put 1, 2 or 3
# ## True ### False ## #
f_resolution = 100     # Spectrum resolution RBW
t_resolution = 0.001   # Time resolution / original 0.001
max_plot_res = 10000
file_name = 'Rec001.csv'
# file_name = 'Rec003.wfm'
fmin = 500
scope_fs = 250000
# scope_fs_new = 25000
# scope_ch_arr = ["RX_out"]
# scope_ch_arr = ["V(out)"]
# fmax = 25000
scope_fs_new = -250000
resample = False
scope_ch_arr = ["Vlrx"]
fmax = 125000
time_string = 'Time'
energy_rise_keys = np.arange(fmin, fmax + f_resolution, f_resolution)
t_start = 0
t_end = 0
cut_samples = 21
cut_samples_2 = 105
path_filter = 'scope'
mixer_arr = np.arange(20, 310, 10)
mixer_arr = [70]
path_output = r'M:\Users\HW Infrastructure\PLC team\ARC\Temp-Eddy'
zero_span_arr = np.arange(100, 25e3, 100)
alpha_filter = 0.2857
down_sample_avg = 1
if not arcs:
    path_folder = r'V:\HW_Infrastructure\Analog_Team\ARC_Data\Results\Venus3\11.4kW DSP (7403495A)\New Arc Detection frequency 02 (24-10-2021)'
    path_dics = ['TX1-RX1', 'TX1-RX2', 'TX1-RX3', 'TX2-RX1', 'TX2-RX2', 'TX2-RX3']
else:
    path_folder = r'V:\HW_Infrastructure\Analog_Team\ARC_Data\Results\Venus3\11.4kW DSP (7403495A)\New Arc Detection frequency 02 (24-10-2021)\Arcs TX2-RX3'
    path_dics = ['Selected arcs for scope FFT']
sliders_num = len(path_dics)
fig_plot_names = ['Energy Rise - Noise floor', 'Energy Rise - Force Telemetries', 'Energy Rise - False Alarms',
                  'Delta - Telemetries and Noise Floor', 'Delta - False Alarms and Noise Floor']
sliders_len = len(fig_plot_names)

if not os.path.exists(path_output):
    os.makedirs(path_output)
if output_text:
    default_stdout = sys.stdout


def main():
    if output_text:
        print()
        print(f'time = {datetime.now()}')
        print()
        sys.stdout = open(f'{path_output}/Terminal Log ({datetime.now().strftime("%d-%m-%Y %H-%M-%S")}).txt', 'w')
    for mixer in mixer_arr:
        if filter_mana[0] and mixer not in filter_mana:
            continue
        fig = make_subplots(rows=1, cols=1, shared_xaxes=False)
        fig_name = f'Energy Rise at Mixer = {mixer}'
        slider_steps = []
        for index_dic, directory in enumerate(path_dics):
            if filter_mana[0] and directory not in filter_mana:
                continue
            # if directory == 'Selected arcs for scope FFT':
            #     path_folder_full = f'{path_folder}\\{directory}\\'
            # else:
            #     path_folder_full = f'{path_folder}\\FAs {directory}\\'
            # file_names = [os.path.splitext(f)[0] for f in os.listdir(path_folder_full) if f.endswith(f'Mixer = {mixer}.csv') and path_filter in f]
            path_folder_full = path_output
            file_names = [os.path.splitext(f)[0] for f in os.listdir(path_folder_full) if file_name in f]
            if print_prints:
                print(f'Got files; length = {len(file_names)}')
            energy_rise_values_all = []
            for index_file, file in enumerate(file_names):
                # if avg_no_energy_rise and file.split(" - ")[1].split(";")[0] != 'Noise floor No Telems':
                #     continue
                # if filter_mana[0] and file.split(" - ")[1].split(";")[0] not in filter_mana:
                if not arcs and filter_mana[0] and int(file.split(" - ")[0].split(" = ")[1]) not in filter_mana:
                    continue
                if print_prints:
                    print()
                    print(f'Loop → index_dic = {index_dic + 1} _ index_file = {index_file + 1}')
                    print(f'file → Mixer = {mixer} _ directory = {directory} _ file = {file}')
                    print()
                # df = NDF_V2.scope_CSV_to_df(path_folder_full, file, scope_ch_arr, scope_ch_arr, False, scope_fs)
                # df = pd.read_csv(path_folder_full + file_name, delimiter='\t')
                df = pd.read_csv(path_folder_full + file_name)
                if df.shape[1] == 3:
                    df.drop('Unnamed: 0', inplace=True, axis=1)
                # del df['Vrxout']
                # df['Time'] = df['Time'] + 1
                # df.to_csv(index=False, sep='\t', header=False, path_or_buf=r'C:\Users\eddy.a\Downloads\Mixer with LT Spice\Tests\Rec003 - In.csv')
                if resample:
                    x = df[time_string].squeeze()
                    y = df[scope_ch_arr].squeeze()
                    f = scipy.interpolate.interp1d(x, y)
                    newx = np.linspace(x.min(), x.max(), int(abs(scope_fs_new) * (x.max() - x.min())))
                    newy = f(newx)
                    df = pd.DataFrame([newx, newy]).T
                    df.columns = [time_string, *scope_ch_arr]
                if scope_fs_new > 0:
                    df_time = df.iloc[::int(scope_fs / scope_fs_new), :].reset_index()
                    # df = df.iloc[::int(scope_fs / scope_fs_new), :].reset_index()
                    df = log_file.avg_no_overlap_list(list(df[scope_ch_arr[0]]), scope_fs, scope_fs_new)
                    df = pd.concat([df_time[time_string], df], axis=1).rename(columns={0: scope_ch_arr[0]})
                    # Fs = round(1 / (df[time_string].iloc[1] - df[time_string].iloc[0]))
                    # df = NDF_V2.df_time_reset(df, time_string)
                    Fs = scope_fs_new
                else:
                    Fs = scope_fs
                if t_start != 0 or t_end != 0:
                    df = NDF_V2.df_Chunk(df, t_start, t_end)
                fft_win = int(Fs / f_resolution)
                fft_win_overlap = int(fft_win - (t_resolution * Fs))
                if fft_win > len(df):
                    fft_win = int(len(df) / 10)
                    fft_win_overlap = int(fft_win * 0.99)
                t, f, z, res_name_arr = NDF_V2.df_stft_RAW_calc(df, Fs, fft_win, fft_win_overlap, scope_ch_arr)
                z = NDF_V2.Z_mag_calc(z)
                if print_prints:
                    print(f'Finished data; spectrogram_EnergyRise_BOTH = {spectrogram_EnergyRise_BOTH}')
                if spectrogram_EnergyRise_BOTH == 1 or spectrogram_EnergyRise_BOTH == 3:
                    # txt = f'Mixer {mixer} @ {directory} - {index_file + 1:02} Spectrogram of{file.split("-")[1].split(";")[0]}'
                    # txt = f'{file} (Mixer = {mixer})'
                    txt = f'{file}'
                    NDF_V2.scpectrogram_plot(z[0], t, f, max_plot_res, fmax, fmin, 0, 0, auto_open_chrome_output, path_output, txt)
                    if print_prints:
                        print(f'Spectrogram HTML has been exported')
                    if spectrogram_EnergyRise_BOTH == 1:
                        continue

                df = NDF_V2.ZeroSpan_calc(z, res_name_arr, t, f, zero_span_arr, scope_ch_arr)
                if directory == 'Selected arcs for scope FFT':
                    file = 'Arcs'
                    directory = 'TX2-RX3'
                else:
                    file = file.split("- ")[1].split(";")[0]
                if print_prints:
                    print(f'Getting the Energy Rise; this will take some time...')
                del z
                energy_rise_values = []
                threads = []
                if index_dic == 0:
                    visible = True
                else:
                    visible = False
                for key, value in df.items():
                    if key == 't':
                        continue
                    elif '@' in key:
                        # freq = int(float(key[6:].split(' @ ')[0]) * 1000)
                        freq = int(float(key.split(' @ ')[1].split(' [')[0]) * 1000)
                        if freq >= fmin:
                            if filter_mana[0]:
                                mana = [filter_mana[-1], filter_mana[-1] + 100, filter_mana[-1] - 100]
                                if freq not in mana:
                                    continue
                            if print_prints:
                                print(f'DFT @ {freq}Hz')
                            if cut_samples_2 != 0:
                                value = value[cut_samples_2:-cut_samples]
                            elif cut_samples != 0:
                                value = value[cut_samples:-cut_samples]
                            if down_sample_avg > 1:
                                value = NDF_V2.Avg_no_overlap(value, down_sample_avg)
                            else:
                                value = list(value)
                            value = log_file.alpha_beta_filter(value, alpha=alpha_filter)
                            import arc_th_calc
                            if avg_no_energy_rise:
                                if frag_factor > 1:
                                    frag_df = []
                                    frag = int(len(value) / frag_factor)
                                    for i in range(0, len(value), frag):
                                        frag_df.append(mean(value[i:i + frag - 1]))
                                    print(f'{mixer},{directory},{file},Fragmented AVG = {frag_factor},{freq},{max(frag_df)}')
                                elif frag_n_sample > 1:
                                    print(f'{mixer},{directory},{file},Rolling AVG = {frag_n_sample},{freq},{max(uniform_filter1d(value, size=frag_n_sample))}')
                                else:
                                    print(f'{mixer},{directory},{file},Whole record AVG,{freq},{mean(value)}')
                            else:
                                if use_plots_no_threads:
                                    energy_rise_values.append(max(NDF_V2.plot_all(value, 20, 15, 12)))
                                else:
                                    x = threading.Thread(target=NDF_V2.energy_rise, args=(mixer, directory, file, freq, value))
                                    x.start()
                                    threads.append(x)
                                    del x
                            del value
                    else:
                        raise ValueError('A very specific bad thing happened.')
                if not use_plots_no_threads:
                    continue
                fig.add_trace(go.Scatter(y=energy_rise_values, x=energy_rise_keys, name=fig_plot_names[index_file],
                                         visible=visible), col=1, row=1)
                energy_rise_values_all.append(energy_rise_values)
                if index_file == 2:
                    fig.add_trace(go.Scatter(y=[a - b for a, b in zip(energy_rise_values_all[1], energy_rise_values_all[0])],
                                             x=energy_rise_keys, name=fig_plot_names[index_file + 1], visible=visible), col=1, row=1)
                    fig.add_trace(go.Scatter(y=[a - b for a, b in zip(energy_rise_values_all[2], energy_rise_values_all[0])],
                                             x=energy_rise_keys, name=fig_plot_names[index_file + 2], visible=visible), col=1, row=1)
            if not use_plots_no_threads or spectrogram_EnergyRise_BOTH == 1:
                continue
            del df, energy_rise_values, energy_rise_values_all
            gc.collect()
            step = dict(args=[{"visible": [False] * sliders_num * sliders_len}, ], label=directory)
            for slider in range(sliders_len):
                step["args"][0]["visible"][index_dic * sliders_len + slider] = True
            slider_steps.append(step)
        if filter_mana[0]:
            continue
        if not use_plots_no_threads:
            for x in threads:
                x.join()
    if output_text:
        sys.stdout.close()
        sys.stdout = default_stdout
        print()
        # print(f'Mixer = {mixer} is FINISHED')
        print(f'time = {datetime.now()}')
        print()
    else:
        if spectrogram_EnergyRise_BOTH == 2 or spectrogram_EnergyRise_BOTH == 3:
            if print_prints:
                print(f'Exporting HTML of Energy Rise')
            fig.update_layout(sliders=[dict(active=0, pad={"t": 50}, steps=slider_steps, bgcolor="#ffb200",
                                            currentvalue=dict(xanchor="center", font=dict(size=16)))])
            fig.update_layout(title=fig_name, title_font_color="#407294", title_font_size=40, legend_title="Plots:")
            plotly.offline.plot(fig, config={'scrollZoom': True, 'editable': True}, filename=f'{path_output}\\{fig_name}.html', auto_open=auto_open_chrome_output)


if __name__ == "__main__":
    main()
