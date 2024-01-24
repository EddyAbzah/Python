import math
import os
import pandas as pd
from io import StringIO

from my_pyplot import omit_plot as _O, plot as _P, print_chrome as _PC, clear as _PP, print_lines as _PL, send_mail as _SM
import Plot_Graphs_with_Sliders as _G
import my_tools


folder = r'M:\Users\Eddy A\Orion\03 D1986 + D1987 (TL)\Majd - telem model orion\KA Link Budget - telem (eddy)'
folder = r'C:\Users\eddy.a\Downloads'
file_filter = ['', '.txt']
file_filter = ['ac-rlossy - Eddy', '.txt']
file_output_suffix = ' - sexy'
conversion = 2          ###   {0: 'OFF', 1: 'Vrms', 2: 'dBm'}
convert_to_volt_rms = lambda a: (10 ** (a / 20)) / (2 * math.sqrt(2))
convert_to_dbm = lambda a: 10 * math.log10((a ** 2) / 0.05)


file_names = [name for name in os.listdir(folder) if all(ext in name for ext in file_filter)]
print(f'len(file_names) = {len(file_names)}')
sex = 1
for file_index, file_name in enumerate(file_names):
    print(f'file_index = {file_index:00}: file_name = {file_name}')
    with open(folder + '\\' + file_name) as file:
        file = file.read()
    file = file.replace('	(', ',').replace('dB,', ',').replace('          ', ',')
    trace_titles = [tt.replace('Freq.', 'Frequency') for tt in file.split('\n')[0].split('\t')]
    print(f'trace_titles = {trace_titles}')
    index_title = trace_titles[0]
    trace_titles = trace_titles[1:]
    if 'Step Information: ' in file:
        file_with_steps = True
        file = file.split('Step Information: ')[1:]
        trace_prefix = [f.split('  (Run: ')[0] + ' - ' for f in file]
        print(f'trace_prefix = {trace_prefix}')
        file = [f[f.find('\n') + 1:] for f in file]
    else:
        file_with_steps = False
        file = [file[file.find('\n') + 1:]]
        trace_prefix = ['']
    print(f'len(file) = {len(file)}')
    all_dfs = pd.DataFrame()

    for index, step in enumerate(file):
        df_names = [trace_prefix[index] + tt for tt in trace_titles]
        df = pd.read_csv(StringIO(step), sep=',', header=None, index_col=0, on_bad_lines='warn').dropna()
        df = df.drop(df.columns[[n * 2 + 1 for n in range(int(len(df.columns) / 2))]], axis=1)
        df.columns = df_names
        df.index.name = index_title

        if conversion > 0:
            df = df.applymap(convert_to_volt_rms)
        if conversion == 2:
            df = df.applymap(convert_to_dbm)
        all_dfs = pd.concat([all_dfs, df], axis=1)
    all_dfs.sort_index(inplace=True)
    print(f'all_dfs.shape = {all_dfs.shape}')
    all_dfs.to_csv(folder + '\\' + file_name[:-4] + file_output_suffix + '.csv')
