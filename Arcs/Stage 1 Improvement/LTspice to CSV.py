## Converting LTspice file to CSVs, but keeping only and the vaules (deleteing phase)

import os
import pandas as pd
from io import StringIO

from my_pyplot import omit_plot as _O, plot as _P, print_chrome as _PC, clear as _PP, print_lines as _PL, send_mail as _SM
import Plot_Graphs_with_Sliders as _G
import my_tools


folder = r'M:\Users\Eddy A\Orion\03 D1986 + D1987 (TL)\Serial relays 06'
file_filter = ['18+50 links - test 50Ω', '.txt']
combine_all_files = True
combined_name = 'sex'

extensionsToCheck = ['.pdf', '.doc', '.xls']
file_names = [name for name in os.listdir(folder) if all(ext in name for ext in file_filter)]
print(f'len(file_names) = {len(file_names)}')
if combine_all_files:
    all_files = pd.DataFrame()
for file_index, file_name in enumerate(file_names):
    print(f'file_index = {file_index:00}: file_name = {file_name}')
    all_dfs = pd.DataFrame()
    with open(folder + '\\' + file_name) as file:
        file = file.read()
    file = file.replace('	(', ',').replace('dB,', ',').split('Step Information: ')
    trace_titles = file[0].split('\n')[0].split('\t')
    print(f'trace_titles = {trace_titles}')
    for step in file[1:]:
        title_prefix = step.split('\n')[0]
        print(f'title_prefix = {title_prefix}')
        df = pd.read_csv(StringIO(step[step.find('\n') + 1:]), sep=',', index_col=0, header=None, on_bad_lines='warn', skipinitialspace=True).dropna()
        df = df.drop([n for n in range(len(df.columns) + 1) if n % 2 == 0 and n != 0], axis=1)
        print(f'df.shape = {df.shape}')
        df.columns = [title_prefix + ' _ ' + title for title in trace_titles[1:]]
        all_dfs = pd.concat([all_dfs, df], axis=1)
    all_dfs.index.name = trace_titles[0]
    if combine_all_files:
        all_dfs.columns = [file_name[:-4] + ' _ ' + title for title in all_dfs.columns]
        all_files = pd.concat([all_files, all_dfs], axis=1)
    else:
        all_dfs.sort_values(0, inplace=True)  # if there is a difference between the x-axes of the dfs
        print(f'all_dfs.shape = {all_dfs.shape}')
        all_dfs.to_csv(folder + '\\' + file_name[:-4] + '.csv')
if combine_all_files:
    try:
        all_files.sort_values(0, inplace=True)  # if there is a difference between the x-axes of the dfs
    except:
        all_files.sort_index(0, inplace=True)
    print(f'all_files.shape = {all_files.shape}')
    all_files.to_csv(folder + '\\' + combined_name + '.csv')
