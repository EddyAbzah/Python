import os
import sys
import scipy
import numpy as np
import pandas as pd
from datetime import datetime
import NDF_V2
import log_file
from my_pyplot import omit_plot as _O, plot as _P, print_chrome as _PC, clear as _PP, print_lines as _PL, send_mail as _SM
import Plot_Graphs_with_Sliders as _G
import my_tools


# ## ### True ##### False ### ## #
output_text = False
auto_open_chrome_output = True
print_prints = True

# ## ### Files ### ## #
path_output = r'M:\Users\HW Infrastructure\PLC team\ARC\Temp-Eddy\LT spice output\No PWM'
path_folder = r'M:\Users\HW Infrastructure\PLC team\ARC\Temp-Eddy\LT spice output\No PWM'
file_extension = '.txt'
filter_file_name = 'resampled - Arc006 OUT (no PWM, 0V instead)'
scope_fs = 50000
scope_ch_arr = ["Sample"]
time_string = 'Time'
delimiter = ','
skiprows = 0

# ## ### Parameters ### ## #
f_resolution = 100     # Spectrum resolution RBW
t_resolution = 0.01   # Time resolution / original 0.001
max_plot_res = 10000
fmin = 500
fmax = 12500
scope_fs_skip = -4
scope_fs_average = -500000
scope_fs_resample = -100000
scope_fs_resample_multiply = 1
slice_and_dice = [False, 2, 1]
t_start = 0
t_end = 0

if not os.path.exists(path_output):
    os.makedirs(path_output)
if output_text:
    default_stdout = sys.stdout


def main():
    if output_text:
        print()
        print(f'main() - start. time = {datetime.now()}')
        print()
        sys.stdout = open(f'{path_output}/Terminal Log ({datetime.now().strftime("%d-%m-%Y %H-%M-%S")}).txt', 'w')
    file_names = [f for f in os.listdir(path_folder) if f.endswith(file_extension) and filter_file_name in f]
    if print_prints:
        print(f'Got files; length = {len(file_names)}')
    for index_file, file in enumerate(file_names):
        if print_prints:
            print(f'index_file = {index_file} _ file = {file}')
        # df = NDF_V2.scope_CSV_to_df(path_folder_full, file, scope_ch_arr, scope_ch_arr, False, scope_fs)
        # df = pd.read_csv(path_folder_full + file_name, delimiter='\t')
        df = pd.read_csv(path_folder + '\\' + file, delimiter=delimiter, skiprows=skiprows, keep_default_na=False)
        Fs = scope_fs
        if df.shape[1] == 3:
            df.drop([s for s in list(df.head(0)) if 'Unnamed' in s][0], inplace=True, axis=1)   # or try del df.iloc[:, 0]
        if slice_and_dice[0]:
            for ind in range(slice_and_dice[1]):
                d = df[int(len(df) / slice_and_dice[1]) * ind:int(len(df) / slice_and_dice[1]) * (ind + 1)]
                if ind == slice_and_dice[2] - 1:
                    df = d
                    del d
                    break
        if scope_fs_skip > 1:
            df = df[::scope_fs_skip].reset_index()
            try:
                del df['index']
            except:
                print("no df['index'] to delete")
            Fs = int(scope_fs / scope_fs_skip)
        if scope_fs_resample > 0:
            x = df.iloc[:, 0].squeeze()
            y = df.iloc[:, 1].squeeze()
            if scope_fs_resample_multiply > 1:
                f = scipy.interpolate.interp1d(x, y)
                newx = np.linspace(x.min(), x.max(), int(abs(scope_fs_resample_multiply) * len(x)))
                newy = f(newx)
                ds_factor = int((1 / (newx[1] - newx[0])) / abs(scope_fs_resample))
                newx = newx[::ds_factor]
                newy = scipy.signal.decimate(newy, ds_factor)
            else:
                f = scipy.interpolate.interp1d(x, y)
                newx = np.linspace(x.min(), x.max(), int(abs(scope_fs_resample) * (x.max() - x.min())))
                newy = f(newx)
            df = pd.DataFrame([newx, newy]).T
            Fs = scope_fs_resample
        df.columns = [time_string, *scope_ch_arr]
        if scope_fs_average > 0:
            df_time = df.iloc[::int(scope_fs / scope_fs_average), :].reset_index()
            df = log_file.avg_no_overlap_list(list(df[scope_ch_arr[0]]), scope_fs, scope_fs_average)
            df = pd.concat([df_time[time_string], df], axis=1).rename(columns={0: scope_ch_arr[0]})
            Fs = scope_fs_average
        if t_start != 0 or t_end != 0:
            df = NDF_V2.df_Chunk(df, t_start, t_end)
        fft_win = int(Fs / f_resolution)
        fft_win_overlap = int(fft_win - (t_resolution * Fs))
        if fft_win > len(df):
            fft_win = int(len(df) / 10)
            fft_win_overlap = int(fft_win * 0.99)
        t, f, z, res_name_arr = NDF_V2.df_stft_RAW_calc(df, Fs, fft_win, fft_win_overlap, scope_ch_arr)
        z = NDF_V2.Z_mag_calc(z)
        NDF_V2.scpectrogram_plot(z[0], t, f, max_plot_res, fmax, fmin, 0, 0, auto_open_chrome_output, path_output, file[:-4])
        del df, t, f, z, res_name_arr
    print()
    print(f'main() - finish. time = {datetime.now()}')
    print()
    if output_text:
        sys.stdout.close()
        sys.stdout = default_stdout


if __name__ == "__main__":
    main()
