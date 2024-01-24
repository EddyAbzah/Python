import os
import pandas as pd
import NDF_V2
from my_pyplot import omit_plot as _O, plot as _P, print_chrome as _PC, clear as _PP, print_lines as _PL, send_mail as _SM
import Plot_Graphs_with_Sliders as _G
import my_tools

# ## ### File parameters ### ## #
path_in = r'M:\Users\LinoyL\Vdc_Criteria_new\MOHAMD_ARCS\RAW_DATA\S1\A1\S1_InvCb_7A\\'
file_in = "S1_7 Rec001 spi 7[A] x3Str Arcer1_InvToCB.txt"
path_out = path_in
file_out = file_in[:-4] + ''
sampling_frequency = 16667
delimiter = ','
set_time = True
data_to_print = 0   # enter column number or a list of columns
skiprows = 0
auto_open_chrome_output = True

# ## ### Spectrogram parameters ### ## #
f_resolution = 50      # Spectrum resolution RBW
t_resolution = 0.01     # Time resolution / original 0.001
max_plot_res = 10000
fmin = 20
fmax = 10000
t_start = 0
t_end = 0

file_names = [f for f in os.listdir(path_in) if file_in in f]
print(f'Length of file_names = {len(file_names)}')
for index_file, file in enumerate(file_names):
    print(f'Getting file number {index_file + 1:00} = {file}')
    df = pd.read_csv(f'{path_in}\\{file_in}', delimiter=delimiter, skiprows=skiprows, keep_default_na=False)
    df.drop(df.filter(regex="Unname"), axis=1, inplace=True)
    if set_time:
        df.set_index(pd.Series(df.index * (1 / sampling_frequency), name='Time'), inplace=True)
    if t_start != 0 or t_end != 0:
        df = NDF_V2.df_Chunk(df, t_start, t_end)
    fft_win = int(sampling_frequency / f_resolution)
    fft_win_overlap = int(fft_win - (t_resolution * sampling_frequency))
    if fft_win > len(df):
        fft_win = int(len(df) / 10)
        fft_win_overlap = int(fft_win * 0.99)
    if type(data_to_print) == int:
        t, f, z, res_name_arr = NDF_V2.df_stft_RAW_calc(df, sampling_frequency, fft_win, fft_win_overlap, [df.columns[data_to_print]])
    else:
        t, f, z, res_name_arr = NDF_V2.df_stft_RAW_calc(df, sampling_frequency, fft_win, fft_win_overlap, data_to_print)
    z = NDF_V2.Z_mag_calc(z)
    NDF_V2.scpectrogram_plot(z[0], t, f, max_plot_res, fmax, fmin, 0, 0, auto_open_chrome_output, path_out, file_out)
    del df, t, f, z, res_name_arr