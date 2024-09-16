import os
import math
import scipy
import numpy as np
import pandas as pd
import log_file
from glob import glob
from my_pyplot import omit_plot as _O, plot as _P, print_chrome as _PC, clear as _PP, print_lines as _PL, send_mail as _SM
import Plot_Graphs_with_Sliders as _G
import my_tools

folder = r'V:\HW_Infrastructure\Analog_Team\ARC_Data\Results\Jupiter\Jupiter+ Improved (7E0872F4-EC)\New frequency detection\LT spice output'
folder_out = r'V:\HW_Infrastructure\Analog_Team\ARC_Data\Results\Jupiter\Jupiter+ Improved (7E0872F4-EC)\New frequency detection\LT spice output resampled (to 16.666K)'
include_subfolders = True
filter_file_name = '0'
file_out_prefix_suffix = ['', '']
file_extension = '.csv'
delimiter = ','
skiprows = 0  # if you output the CSV via Web, there will be 20 rows of nonsense
text_filter = (True, ['inf'])
scope_fs_skip = -5
scope_fs_average = -50000
scope_fs_resample = 16666
scope_fs_resample_multiply = -2
columns_titles = (False, ['Time', 'Sample'])

if include_subfolders:
    file_names = glob(folder + "/*/*", recursive=True)
else:
    file_names = glob(folder + "/*", recursive=True)
print(f'Got files; length = {len(file_names)}')
for index_file_in, file_in_init in enumerate(file_names):
    file_in = os.path.basename(file_in_init)
    file_out = file_out_prefix_suffix[0] + file_in + file_out_prefix_suffix[1]
    if text_filter[0]:
        file = open(file_in_init, "r").readlines()
        for index, line in enumerate(file):
            if any(text in line for text in text_filter[1]):
                print(f'Text found in line = {index}')
                exit(666)
        print('Text seems good')
        del file
    df = pd.read_csv(file_in_init, delimiter=delimiter, skiprows=skiprows)
    x = list(df.iloc[:, 0].squeeze())
    y = (df.iloc[:, 1].squeeze())
    time_start = x[0]
    time_stop = x[-1]
    scope_fs = len(x) / (x[-1] - x[0])
    if columns_titles[0]:
        columns_titles[1] = list(df.head(0))
    print(f'DataFrame is ready; Fs = {scope_fs}, t-start = {time_start} and t-stop = {time_stop}')

    if scope_fs_skip > 0:
        x = x[::scope_fs_skip]
        y = y[::scope_fs_skip]
        scope_fs = int(scope_fs / scope_fs_skip)
    if scope_fs_resample > 0:
        f = scipy.interpolate.interp1d(x, y)
        time_delta = time_stop - time_start
        if scope_fs_resample_multiply > 1:
            resample_len = int(round(scope_fs_resample_multiply * len(x) / (scope_fs_resample * time_delta))) * scope_fs_resample * time_delta
            x = np.linspace(time_start, time_stop, int(resample_len) + 1)
            y = f(x)
            ds_factor = int(round(round(1 / (x[1] - x[0])) / scope_fs_resample))
            x = x[::ds_factor].round(decimals=len(str(math.ceil(scope_fs_resample))))
            y = scipy.signal.decimate(y, ds_factor)
        else:
            x = np.linspace(time_start, time_stop, int(scope_fs_resample * time_delta) + 1).round(decimals=len(str(math.ceil(scope_fs_resample))))
            y = f(x)
        scope_fs = scope_fs_resample
    if scope_fs_average > 0:
        x = x[::int(scope_fs / scope_fs_average)]
        y = log_file.avg_no_overlap_list(y, scope_fs, scope_fs_average, convert_to_pd_df=False)
        scope_fs = scope_fs_average

    print(f'Creating CSV of shape = {df.shape}; Final fs = {scope_fs}')
    df = pd.DataFrame([x, y]).T
    df.columns = columns_titles[1]
    df.to_csv(folder_out + '\\' + file_out, index=False)
