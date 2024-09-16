## Imports and main lists:
import plotly.graph_objects as go
import bottleneck as bn
import pandas as pd
import numpy as np
import log_file
import scipy
import arc_th_calc
import Goertzel
import plotly
import os
import sys
from datetime import datetime
from plotly.subplots import make_subplots
from my_pyplot import omit_plot as _O, plot as _P, print_chrome as _PC, clear as _PP, print_lines as _PL, send_mail as _SM
import Plot_Graphs_with_Sliders as _G
import my_tools
mana = []
sex = []


## Variables
filter_sex = ['Selected arcs for scope FFT', 'Noise floor No Telems', 'Force Telemetries discrete', 'False Alarms case 9']
# filter_mana = [True, None, 140, filter_sex[1], 6200]
# arcs = False
filter_mana = [False, 'TX2-RX3', 140, filter_sex[0], 6000]
arcs = False
ltspice_record = True
# region MANA
remove_spb_avg = True
add_hamming_window = True
use_ker_not_sin_cos = False
add_goertzel_meas = True
arcs_start_at = 340
output_text = False
print_prints = False
# ## True ### False ## #
scope_fs = 50e3
t_start = 0
t_end = 0
# mixer_arr = np.arange(20, 310, 10)
mixer_arr = [70]
# fif_arr = np.concatenate((np.arange(750, 2000, 50), np.arange(2000, 7500, 25), np.arange(7500, 10000, 50), np.arange(10000, 26000, 100)), axis=0)
fif_arr = [6000]
path_output = r'C:\Users\eddy.a\Downloads\Mixer with LT Spice\Tests 02'
zero_span_arr = np.arange(100, 25e3, 100)
alpha_filter = 0.2857
resample = True
down_sample_cut = 4
spb = 357
add_vdc = 0
if not arcs:
    path_folder = r'V:\HW_Infrastructure\Analog_Team\ARC_Data\Results\Venus3\11.4kW DSP (7403495A)\New Arc Detection frequency 02 (24-10-2021)'
    path_dics = ['TX1-RX1', 'TX1-RX2', 'TX1-RX3', 'TX2-RX1', 'TX2-RX2', 'TX2-RX3']
else:
    path_folder = r'V:\HW_Infrastructure\Analog_Team\ARC_Data\Results\Venus3\11.4kW DSP (7403495A)\New Arc Detection frequency 02 (24-10-2021)\Arcs TX2-RX3'
    path_dics = ['Selected arcs for scope FFT']
if ltspice_record:
    path_folder = r'C:\Users\eddy.a\Downloads\Mixer with LT Spice'
    path_dics = ['Tests 02']
sliders_num = len(path_dics)
fig_plot_names = ['Energy Rise - Noise floor', 'Energy Rise - Force Telemetries', 'Energy Rise - False Alarms',
                  'Delta - Telemetries and Noise Floor', 'Delta - False Alarms and Noise Floor']
sliders_len = len(fig_plot_names)
if not os.path.exists(path_output):
    os.makedirs(path_output)
scope_ch_arr = ["CH4"]
if output_text:
    default_stdout = sys.stdout
# endregion
## end of Variables


def main():
    ## Main:
    # region main()
    print()
    print('Script start')
    print(f'time = {datetime.now()}')
    print()
    for index_dic, dic in enumerate(path_dics):
        if output_text:
            sys.stdout = open(f'{path_output}/Terminal Log ({datetime.now().strftime("%d-%m-%Y %H-%M-%S")}).txt', 'w')
        all_data = []
        if index_dic == 0:
            if arcs:
                if add_goertzel_meas:
                    print('Pairing,Test type,Mixer,Detection,Rise Arc (embedded), Rise other,Rise delta,Rise Arc (Goertzel), Rise other,Rise delta')
                else:
                    print('Pairing,Test type,Mixer,Detection,Energy Rise')
            else:
                if add_goertzel_meas:
                    print('Pairing,Test type,Mixer,Detection,Energy Rise Arc,Energy Rise other,Energy Rise Delta')
                else:
                    print('Pairing,Test type,Mixer,Detection,Energy Rise')
        if filter_mana[0] and filter_mana[1] is not None and dic not in filter_mana:
            continue
        for mixer in mixer_arr:
            if filter_mana[0] and filter_mana[2] is not None and mixer != filter_mana[2]:
                continue
            if print_prints:
                print(f'mixer = {mixer}')
            fig_name = f'dic = {dic} & mixer = {mixer}'
            if print_prints:
                print(fig_name)
            if arcs:
                full_path = f'{path_folder}/{dic}'
            else:
                full_path = f'{path_folder}/Non-Arcs {dic}'
            if ltspice_record:
                file_names = [r'C:\Users\eddy.a\Downloads\Mixer with LT Spice\Tests 02\Rec001.csv']
            else:
                file_names = [f for f in os.listdir(full_path) if f.endswith(f'Mixer = {mixer}.csv') and 'scope' in f]
            if print_prints:
                print(f'Got files; length = {len(file_names)}')
            for index_file, file in enumerate(file_names):
                if ltspice_record:
                    test_type = 'ltspice_record'
                else:
                    if arcs:
                        test_type = filter_mana[3]
                    else:
                        test_type = file.split(' - ')[1].split('; ')[0]
                if filter_mana[0] and filter_mana[3] is not None and test_type != filter_mana[3]:
                    continue
                if ltspice_record:
                    df = pd.read_csv(file)
                else:
                    df = pd.read_csv(f'{full_path}/{file}').dropna(how='all', axis='columns')[scope_ch_arr[0]]
                if ltspice_record and resample:
                    x = df.iloc[:, 1].squeeze()
                    y = df.iloc[:, 2].squeeze()
                    f = scipy.interpolate.interp1d(x, y)
                    newx = np.linspace(x.min(), x.max(), int(abs(scope_fs) * (x.max() - x.min())))
                    newy = f(newx)
                    df = pd.DataFrame(newy).squeeze()
                if add_vdc != 0:
                    df = df + add_vdc
                if down_sample_cut != 0:
                    df = df[::down_sample_cut].reset_index(drop=True)
                if remove_spb_avg:
                    df = df - bn.move_mean(df, window=spb, min_count=1)
                if add_hamming_window:
                    hamming_window = [0.54 - 0.46 * np.cos(2 * np.pi * n / spb) for n in range(spb)]

                for fif in fif_arr:
                    if filter_mana[0] and filter_mana[4] is not None and fif != filter_mana[4]:
                        continue
                    if print_prints:
                        print(f'fif = {fif}')
                    if use_ker_not_sin_cos:
                        ker = [np.exp((-1j * 2 * np.pi * fif * n) / (scope_fs / 4)) for n in range(spb)]
                    else:
                        sin = [np.sin((2 * np.pi * fif * n) / (scope_fs / 4)) for n in range(spb)]
                        cos = [np.cos((2 * np.pi * fif * n) / (scope_fs / 4)) for n in range(spb)]
                    if add_goertzel_meas:
                        goertzel_dft_after_sum = Goertzel.GoertzelSampleBySample_func(df, fif, scope_fs / 4, spb)

                    dft_after_sum = []
                    for window in [df[i:i + spb] for i in range(0, len(df), spb)]:
                        if add_hamming_window:
                            window = [a * b for a, b in zip(window, hamming_window)]
                        if use_ker_not_sin_cos:
                            real = np.real([a * b for a, b in zip(ker, window)])
                            img = np.imag([a * b for a, b in zip(ker, window)])
                        else:
                            real = [a * b for a, b in zip(sin, window)]
                            img = [a * b for a, b in zip(cos, window)]
                        dft_after_sum.append(sum(real) ** 2 + sum(img) ** 2)

                    if print_prints:
                        print(f'starting alpha filter')
                    if alpha_filter < 1:
                        dft_after_filter = log_file.alpha_beta_filter(dft_after_sum, alpha=alpha_filter)
                        if add_goertzel_meas:
                            goertzel_dft_after_filter = log_file.alpha_beta_filter(goertzel_dft_after_sum, alpha=alpha_filter)
                    if t_start != 0:
                        if arcs and mixer == 190:
                            dft_after_filter = dft_after_filter[t_start + 137:]
                        else:
                            dft_after_filter = dft_after_filter[t_start:]
                        if add_goertzel_meas:
                            goertzel_dft_after_filter = goertzel_dft_after_filter[t_start:]
                    if t_end != 0:
                        dft_after_filter = dft_after_filter[:-t_end]
                        if add_goertzel_meas:
                            goertzel_dft_after_filter = goertzel_dft_after_filter[:-t_end]
                    dft_db = [10 * np.log10(n) for n in dft_after_filter]
                    mana.append(dft_db)
                    sex.append(f'dic = {dic},   test_type = {test_type},   mixer = {mixer},   fif = {fif}')

                    if arcs:
                        temp_rise = arc_th_calc.plot_all(dft_db, window_size=20, filter_size=15, over_th_limit=12)
                        rise = max(temp_rise[:arcs_start_at])
                        rise_2 = max(temp_rise[arcs_start_at:])
                    else:
                        rise = max(arc_th_calc.plot_all(dft_db, window_size=20, filter_size=15, over_th_limit=12))
                    if add_goertzel_meas:
                        goertzel_dft_db = [10 * np.log10(n) for n in goertzel_dft_after_filter]
                        if arcs:
                            temp_rise = arc_th_calc.plot_all(goertzel_dft_db, window_size=20, filter_size=15, over_th_limit=12)
                            goertzel_rise = max(temp_rise[:arcs_start_at])
                            goertzel_rise_2 = max(temp_rise[arcs_start_at:])
                            all_data.append(f'{dic},{test_type},{mixer},{fif},{rise_2},{rise},{rise_2 - rise},{goertzel_rise_2},{goertzel_rise},{goertzel_rise_2 - goertzel_rise}')
                        else:
                            goertzel_rise = max(arc_th_calc.plot_all(goertzel_dft_db, window_size=20, filter_size=15, over_th_limit=12))
                            all_data.append(f'{dic},{test_type},{mixer},{fif},{rise},{goertzel_rise}')

                    else:
                        if arcs:
                            all_data.append(f'{dic},{test_type},{mixer},{fif},{rise_2},{rise},{rise_2 - rise}')
                        else:
                            all_data.append(f'{dic},{test_type},{mixer},{fif},{rise}')
                    if print_prints:
                        print(f'dic = {dic}, mixer = {mixer}, test_type = {test_type}')
        for string in all_data:
            print(string)
        if output_text:
            sys.stdout.close()
            sys.stdout = default_stdout
        print()
        print(f'Mixer = {mixer} is FINISHED')
        print(f'time = {datetime.now()}')
        print()
    # endregion
## end of Main


if __name__ == "__main__":
    main()


def print_all_pp():
    ## Print PP:
    for info, trace in zip (sex, mana):
        print(f'Printing {info}')
        _P(trace, f'{" ".join([s.split(" = ")[1] for s in sex[0].split(",   ")[::2]])} @ {info.split(",   ")[-1].split(" = ")[1]}')
## end of Print PP


def print_all_chrome():
    ## Print Chrome:
    fig = make_subplots(rows=1, cols=1, shared_xaxes=False)
    path = r'M:\Users\HW Infrastructure\PLC team\ARC\Temp-Eddy\print_all_chrome.html'
    for index, (info, trace) in enumerate(zip(sex, mana)):
        name = f'{" ".join([s.split(" = ")[1] for s in info.split(",   ")[::2]])} @ {info.split(",   ")[-1].split(" = ")[1]}'
        print(f'Printing {info}')
        fig.add_trace(go.Scatter(y=trace, name=name), col=1, row=1)
    fig.update_layout(title='print_all_chrome', title_font_color="#407294", title_font_size=40, legend_title="Plots:")
    plotly.offline.plot(fig, config={'scrollZoom': True, 'editable': True}, filename=path, auto_open=True)
## end of Print PP Chrome
