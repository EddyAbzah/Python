import os
import math
import scipy
import numpy as np
import pandas as pd
import log_file
# from my_pyplot import plot as _P, clear as _PP

folder = r'M:\Users\HW Infrastructure\PLC team\ARC\Temp-Eddy\LT spice output\No PWM'
filter_file_name = '0'
file_out_prefix_suffix = ['resampled - ', '']
file_extension = '.txt'
delimiter = '\t'
skiprows = 0  # if you output the CSV via Web, there will be 20 rows of nonsense
text_filter = (True, ['inf'])
scope_fs_skip = -5
scope_fs_average = -50000
scope_fs_resample = 50000
scope_fs_resample_multiply = -2
columns_titles = (False, ['Time', 'Sample'])

file_names = [f for f in os.listdir(folder) if f.endswith(file_extension) and f'{filter_file_name}' in f]
print(f'Got files; length = {len(file_names)}')
for index_file_in, file_in in enumerate(file_names):
    file_out = file_out_prefix_suffix[0] + file_in + file_out_prefix_suffix[1]
    if text_filter[0]:
        file = open(folder + '\\' + file_in, "r").readlines()
        for index, line in enumerate(file):
            if any(text in line for text in text_filter[1]):
                print(f'Text found in line = {index}')
                exit()
        print('Text seems good')
        del file
    df = pd.read_csv(folder + '\\' + file_in, delimiter=delimiter, skiprows=skiprows)
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
    df.to_csv(folder + '\\' + file_out, index=False)
