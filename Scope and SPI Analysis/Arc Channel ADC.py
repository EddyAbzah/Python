## Imports and main lists:
import bottleneck as bn
import pandas as pd
import numpy as np
import log_file
import scipy
import arc_th_calc
import Goertzel
import os
import sys
from statistics import mean
from datetime import datetime

from my_pyplot import omit_plot as _O, plot as _P, print_chrome as _PC, clear as _PP, print_lines as _PL, send_mail as _SM
import Plot_Graphs_with_Sliders as _G
import my_tools

all_dfs = []
all_strings = []
all_rises = []


## Variables:
output_text = False
chrome_plot__auto_open = [True, True]
print_prints = True
remove_spb_avg__size = [True, 65535]   # = spb or UInt16.MaxValue (65535)
add_hamming_window = True
add_energy_rise_plot = False
rise_calc_cut = 0
use_ker_not_sin_cos = True
add_goertzel_meas = False
arcs = False
arcs_start_at = 150
arcs_stop_at = 233
scope_fs_init = 16667
spb = 476  # 357 for Venus, 476 for Jupiter
down_sample_cut__filter = [-4, '12.5K']
alpha_filter = 0.2857
add_vdc = 0
sample_start = 0
t_start = 0
t_end = 0


## Paths and filters:
path_output = r'M:\Users\LinoyL\DATA_ARCS\ALL_DATA\NewDATA_MSP\S1\A2A3\S1_ShortLong_3A'
path_folder = r'M:\Users\LinoyL\DATA_ARCS\ALL_DATA\NewDATA_MSP\S1\A2A3\S1_ShortLong_3A'
include_subfolders = False
filter_files = [True, 3]
file_extension = '.txt'
filter_file_name = ''
scope_ch_arr = ['RX_out,']
# scope_ch_arr = 2
time_string = 'Time'
delimiter = ','
fif_arr = np.concatenate((np.arange(750, 2000, 50), np.arange(2000, 7500, 25), np.arange(7500, 10050, 50)), axis=0)
fif_arr = [6000]
send_mail_when_finished = False
if not os.path.exists(path_output):
    os.makedirs(path_output)
if output_text:
    default_stdout = sys.stdout


print(f'main() - start. time = {datetime.now()}')
if include_subfolders:
    file_names = [os.path.join(root, f) for root, dirs, files in os.walk(path_folder) for f in files if
                  f.endswith(file_extension) and f'{filter_file_name}' in f]
else:
    file_names = [f for f in os.listdir(path_folder) if f.endswith(file_extension) and f'{filter_file_name}' in f]
if filter_files[0]:
    file_names = [file_names[filter_files[1]]]
print(f'Got files; length = {len(file_names)}')
if output_text:
    sys.stdout = open(f'{path_output}/Terminal Log ({datetime.now().strftime("%d-%m-%Y %H-%M-%S")}).txt', 'w')
    print(f'Got files; length = {len(file_names)}')
for index_file, file in enumerate(file_names):
    if type(scope_ch_arr) == int:
        df = pd.read_csv(path_folder + '\\' + file, delimiter=delimiter).dropna(how='all', axis='columns').iloc[:, scope_ch_arr]
    else:
        df = pd.read_csv(path_folder + '\\' + file, delimiter=delimiter).dropna(how='all', axis='columns')[scope_ch_arr[0]]
    df = df[sample_start:]
    if add_vdc != 0:
        df = df + add_vdc
    if down_sample_cut__filter[0] > 0 and down_sample_cut__filter[1] in file:
        df = df[::down_sample_cut__filter[0]].reset_index(drop=True)
        scope_fs = int(scope_fs_init / down_sample_cut__filter[0])
    else:
        scope_fs = scope_fs_init
    if remove_spb_avg__size[0]:
        df = df - bn.move_mean(df, window=min(remove_spb_avg__size[1], len(df)), min_count=1)
    if add_hamming_window:
        hamming_window = [0.54 - 0.46 * np.cos(2 * np.pi * n / spb) for n in range(spb)]

    for fif in fif_arr:
        if print_prints:
            print(f'fif = {fif}')
        if use_ker_not_sin_cos:
            ker = [np.exp((-1j * 2 * np.pi * fif * n) / scope_fs) for n in range(spb)]
        else:
            cos = [np.cos((2 * np.pi * fif * n) / scope_fs) for n in range(spb)]
            sin = [np.sin((2 * np.pi * fif * n) / scope_fs) for n in range(spb)]
        if add_goertzel_meas:
            goertzel_dft_after_sum = Goertzel.GoertzelSampleBySample_func(df, fif, scope_fs, spb)

        dft_after_sum = []
        for window in [df[i:i + spb] for i in range(0, len(df), spb)]:
            if add_hamming_window:
                window = [a * b for a, b in zip(window, hamming_window)]
            if use_ker_not_sin_cos:
                real = np.real([a * b for a, b in zip(ker, window)])
                img = np.imag([a * b for a, b in zip(ker, window)])
            else:
                real = [a * b for a, b in zip(cos, window)]
                img = [a * b for a, b in zip(sin, window)]
            dft_after_sum.append(sum(real) ** 2 + sum(img) ** 2)

        if print_prints:
            print(f'starting alpha filter')
        if alpha_filter < 1:
            dft_after_filter = log_file.alpha_beta_filter(dft_after_sum, alpha=alpha_filter)
            if add_goertzel_meas:
                goertzel_dft_after_filter = log_file.alpha_beta_filter(goertzel_dft_after_sum, alpha=alpha_filter)
        if t_start != 0:
            dft_after_filter = dft_after_filter[t_start:]
            if add_goertzel_meas:
                goertzel_dft_after_filter = goertzel_dft_after_filter[t_start:]
        if t_end != 0:
            dft_after_filter = dft_after_filter[:-t_end]
            if add_goertzel_meas:
                goertzel_dft_after_filter = goertzel_dft_after_filter[:-t_end]
        dft_db = [10 * np.log10(n) for n in dft_after_filter]
        all_dfs.append(dft_db)
        if add_goertzel_meas:
            goertzel_dft_db = [10 * np.log10(n) for n in goertzel_dft_after_filter]
            all_dfs.append(goertzel_dft_db)
            goertzel_temp_string = f'file = {file}, fif = {fif}'
        temp_string = f'file = {file}, fif = {fif}'
        # temp_string = f'{file.split(")")[0]}) {file.split(" - ")[1].split(".")[0]} @ {int(fif)}Hz'
        if arcs:
            temp_rise = arc_th_calc.plot_all(dft_db, window_size=20, filter_size=15, over_th_limit=12)[rise_calc_cut:]
            all_rises.append(temp_rise)
            rise = max(temp_rise[:arcs_start_at])
            rise_2 = max(temp_rise[arcs_start_at:arcs_stop_at])
            temp_string = f'{temp_string}, rise[dB] = {rise}, rise_2[dB] = {rise_2}'
        else:
            rise = arc_th_calc.plot_all(dft_db, window_size=20, filter_size=15, over_th_limit=12)[rise_calc_cut:]
            all_rises.append(rise)
            rise = max(rise)
            temp_string = f'{temp_string}, rise[dB] = {rise}'
        if add_goertzel_meas:
            if arcs:
                temp_rise = arc_th_calc.plot_all(goertzel_dft_db, window_size=20, filter_size=15, over_th_limit=12)[rise_calc_cut:]
                all_rises.append(temp_rise)
                goertzel_rise = max(temp_rise[:arcs_start_at])
                goertzel_rise_2 = max(temp_rise[arcs_start_at:arcs_stop_at])
                goertzel_temp_string = f'{goertzel_temp_string}, goertzel_rise = {goertzel_rise}, goertzel_rise_2 = {goertzel_rise_2}'
            else:
                goertzel_rise = arc_th_calc.plot_all(goertzel_dft_db, window_size=20, filter_size=15, over_th_limit=12)[rise_calc_cut:]
                all_rises.append(goertzel_rise)
                goertzel_rise = max(goertzel_rise)
                goertzel_temp_string = f'{goertzel_temp_string}, goertzel_rise = {goertzel_rise}'
        if arcs:
            temp_string = f'{temp_string}, NF1-MIN[dB] = {min(dft_db[:arcs_start_at])}, NF1-MAX[dB] = {max(dft_db[:arcs_start_at])}, NF1-AVG[dB] = {mean(dft_db[:arcs_start_at])}'
            temp_string = f'{temp_string}, NF2-MIN[dB] = {min(dft_db[arcs_start_at:arcs_stop_at])}, NF2-MAX[dB] = {max(dft_db[arcs_start_at:arcs_stop_at])}, NF2-AVG[dB] = {mean(dft_db[arcs_start_at:arcs_stop_at])}'
        else:
            temp_string = f'{temp_string}, NF-MIN[dB] = {min(dft_db)}, NF-MAX[dB] = {max(dft_db)}, NF-AVG[dB] = {mean(dft_db)}'
        all_strings.append(temp_string)
        if add_goertzel_meas:
            all_strings.append(temp_string)
    if index_file > 2:
        break

for index, string in enumerate(all_strings):
    print(f'{string}, len(df) = {len(all_dfs[index])}')
if chrome_plot__auto_open[0]:
    if add_energy_rise_plot:
        _PC([*all_dfs, *all_rises], labels=[*all_strings, *all_strings], path=path_output, file_name=filter_file_name, title=filter_file_name, auto_open=chrome_plot__auto_open[1])
    else:
        _PC(all_dfs, labels=all_strings, path=path_output, file_name=filter_file_name, title=filter_file_name, auto_open=chrome_plot__auto_open[1])
print(f'main() - finish. time = {datetime.now()}')
if output_text:
    sys.stdout.close()
    sys.stdout = default_stdout
    print(f'main() - finish. time = {datetime.now()}')
if send_mail_when_finished:
    _SM(pc='eddyab-pc', file='Arc Channel ADC.py')
