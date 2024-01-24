## Imports and main lists:
import pandas as pd
import numpy as np
import log_file
import scipy
import os
import sys
import log_file
import arc_th_calc
from my_pyplot import omit_plot as _O, plot as _P, print_chrome as _PC, clear as _PP, print_lines as _PL, send_mail as _SM
import Plot_Graphs_with_Sliders as _G
import my_tools


## True or False
chrome_plot__auto_open = [False, True]
remove_zeros = True
remove_last_negatives = False
down_sample = [False, 357]
cut_last_samples = [True, 3]
calculate_vdc_th = True
print_all = False


## Paths and filters
path_output = r'M:\Users\HW Infrastructure\PLC team\ARC\Temp-Eddy\V-DC'
path_folder = r'M:\Users\HW Infrastructure\PLC team\ARC\Temp-Eddy\V-DC\Arcs with FA v09 test 02 (28-02-2022)'
file_extension = '.txt'
filter_file_name = 'spi'
delimiter = ','
if not os.path.exists(path_output):
    os.makedirs(path_output)


def MANA(df, alpha):
    list_2 = {}
    list_2['VD'] = df['Voltage Drop']
    list_2['VDC'] = df['V-DC']
    excel_len = 400
    c = int(len(df) / excel_len)
    list_2['AF'] = alpha
    list_2['AB'] = log_file.alpha_beta_filter(list_2['VDC'], alpha)
    list_2['A'] = log_file.avg_no_overlap(log_file.convert_to_df(list_2['AB']), c)
    Ks, Filters = arc_th_calc.voltage_algo(list_2['A'], 20, 15, 20, True, 0.05)
    cut = 40
    list_2['F'] = Filters[cut:]
    list_2['K'] = Ks[cut:]
    return list_2


## Main
file_names = [f for f in os.listdir(path_folder) if f.endswith(file_extension) and f'{filter_file_name}' in f]
print(f'Got files; length = {len(file_names)}')
if calculate_vdc_th:
    sex = []
    s = []
    ss = []
    sss = []
    ssss = []
for index_file, file in enumerate(file_names):
    df = pd.read_csv(path_folder + '\\' + file, delimiter=delimiter).dropna(how='all', axis='columns')
    if calculate_vdc_th:
        for mana in [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009]:
            sex.append(MANA(df, mana))
            s.append(sex[-1]['A'])
            ss.append(f'Alpha = {sex[-1]["AF"]} - Vdc')
            sss.append(sex[-1]['K'])
            ssss.append(f'Alpha = {sex[-1]["AF"]} - Ks')
        break
_PC(s, labels=ss, path=r'M:\Users\HW Infrastructure\PLC team\ARC\Temp-Eddy', file_name='Rec001 SPI alphas',
        title='SPI alphas')
_PC(sss, labels=ssss, path=r'M:\Users\HW Infrastructure\PLC team\ARC\Temp-Eddy', file_name='Rec001 SPI Ks (TH = 0.05)',
        title='SPI Ks (TH = 0.05)')
    # if print_all:
    #     _PC(df, path=path_folder, file_name=file[:-4], title=file)
    #     continue
    # if remove_zeros:
    #     df = df.loc[(df != 0).any(1)]
    # if remove_last_negatives:
    #     cut_at = df.loc[(df < 0).any(1)].index[0]
    #     df = df[:cut_at]
    # if down_sample[0]:
    #     df = df[::down_sample[1]]
    # if cut_last_samples[0]:
    #     df = df[:-cut_last_samples[1]]
    # del df['SPI Sample']
    # if chrome_plot__auto_open[0]:
    #     _PC(df, file_name=file, title=file, path=path_output, auto_open=chrome_plot__auto_open[1])
