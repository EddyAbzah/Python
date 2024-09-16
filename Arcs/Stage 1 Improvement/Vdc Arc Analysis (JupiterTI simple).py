import os
import pandas as pd
import arc_th_calc
import log_file
from my_pyplot import omit_plot as _O, plot as _P, print_chrome as _PC, clear as _PP, print_lines as _PL, send_mail as _SM
import Plot_Graphs_with_Sliders as _G
import my_tools

# TempSPI = True
TempSPI = False

if TempSPI:
    path_in = r'M:\Users\HW Infrastructure\PLC team\ARC\Temp-SPI'
    path_out = path_in
    filter_name = ['.txt', 'record_']
else:
    path_in = r'M:\Users\HW Infrastructure\PLC team\ARC\Temp-Eddy\Digital_Record 08_02_23 18_22'
    path_out = path_in  # or path_in + '\\Graphs'
    filter_name = ['.csv', 'spi']
# filter_name.append('002')
pd_columns = [True, ['Case 19', 'Case 62', 'Case 583', 'Case 42', 'Case 159', 'Case 160']]
pd_columns = [True, ['Vdc fast (19)', 'Vdc zero-cross (62)', 'Vdc pre-buffer (159)', 'Vdc buffer (160)', 'L1 Iac (39)', 'L1 PF (162)']]
title_pwr = 'Case 160(0.0625)'
titles = [pd_columns[1][0], pd_columns[1][4], pd_columns[1][5]]
# alpha_filter = [0.0001, 0.000075, 0.00005]
alpha_filter = [0.0001]
sample_rate_spi = 16667
sample_rate_vdc = 140   # min
sample_rate_pwr = 35
test_type = [1, ['6 sample AVG', '10 sample Moving AVG']]
cut_record = [-2, 687, 872, 874]    # first bit = record number non-zero
Ks_not_float = [True, 0.1]
_G.change_path(path_out)
_G.change_action(new_kill_chrome=False, new_auto_open_chrome_output=True, new_shared_x_axis=True, new_slider_prefix='')


for file_index, file_name in enumerate([f for f in os.listdir(path_in) if all([x in f.lower() for x in filter_name])]):
    temp_csv = pd.read_csv(path_in + '\\' + file_name)
    if pd_columns[0]:
        temp_csv.columns = pd_columns[1]
    temp_vdc1 = temp_csv[titles[0]]
    temp_vdc1_list = []
    temp_vdc_th_list1 = []
    for alpha in alpha_filter:
        print(f'Vdc1 => Filtering the SPI')
        temp_vdc1_list.append(pd.Series(log_file.alpha_beta_filter(temp_vdc1, alpha)))
        print(f'Vdc1 => Downsampling')
        temp_vdc1_list[-1] = pd.Series(log_file.avg_no_overlap(temp_vdc1_list[-1], sample_rate_spi, sample_rate_pwr))
        if file_index == cut_record[0] - 1:
            temp_vdc1_list[-1] = temp_vdc1_list[-1][:cut_record[1]]
        print(f'Vdc1 => Calculating the TH')
        if Ks_not_float[0]:
            temp_vdc_th_list1.append(arc_th_calc.voltage_drop_algo2_Ks(temp_vdc1_list[-1], Ks_not_float[1], window_size=20, filter_size=15))
        else:
            temp_vdc_th_list1.append(arc_th_calc.voltage_drop_algo2(temp_vdc1_list[-1], window_size=20, filter_size=15, over_th_limit=12))

    temp_vdc2 = temp_csv[titles[1]]
    print(f'Vdc2 => Dropping consecutive duplicates')
    # temp_vdc2 = temp_vdc2.loc[temp_vdc2.shift() != temp_vdc2]
    # temp_vdc2 = temp_vdc2.loc[::int(sample_rate_spi / sample_rate_vdc)]
    temp_vdc2 = log_file.remove_consecutive_duplicates(temp_vdc2, int(sample_rate_spi / sample_rate_vdc))
    print(f'Vdc2 => Downsampling method')
    if test_type[0] == 0:
        temp_vdc2 = pd.Series(log_file.avg_no_overlap(temp_vdc2, 6))
    elif test_type[0] == 1:
        temp_vdc2 = pd.Series(log_file.avg_with_overlap(temp_vdc2, 4, 6))
    if file_index == cut_record[0] - 1:
        temp_vdc2 = temp_vdc2[:cut_record[2]]
    print(f'Vdc2 => Calculating the TH')
    if Ks_not_float[0]:
        temp_vdc_th_list2 = arc_th_calc.voltage_drop_algo2_Ks(temp_vdc2, Ks_not_float[1], window_size=20, filter_size=15)
    else:
        temp_vdc_th_list2 = arc_th_calc.voltage_drop_algo2(temp_vdc2, window_size=20, filter_size=15, over_th_limit=12)

    temp_vdc3 = temp_csv[titles[2]]
    print(f'Vdc3 => Dropping consecutive duplicates')
    # temp_vdc3 = temp_vdc3.loc[temp_vdc3.shift() != temp_vdc3]
    # temp_vdc3 = temp_vdc3.loc[::int(sample_rate_spi / sample_rate_vdc)]
    temp_vdc3 = log_file.remove_consecutive_duplicates(temp_vdc3, int(sample_rate_spi / sample_rate_pwr))
    print(f'Vdc3 => Downsampling method')
    if file_index == cut_record[0] - 1:
        temp_vdc3 = temp_vdc3[:cut_record[3]]
    print(f'Vdc3 => Calculating the TH')
    if Ks_not_float[0]:
        temp_vdc_th_list3 = arc_th_calc.voltage_drop_algo2_Ks(temp_vdc3, Ks_not_float[1], window_size=20, filter_size=15)
    else:
        temp_vdc_th_list3 = arc_th_calc.voltage_drop_algo2(temp_vdc3, window_size=20, filter_size=15, over_th_limit=12)

    # _G.add_data(temp_vdc1_list, [f'{titles[0][:titles[0].find("(")]} (Moving AVG = {s + 4})' for s in alpha_filter], true_if_matrix=True, title_matrix=f'{titles[0][:titles[0].find("(")]} (Moving AVG = {", ".join(map(lambda x: str(x + 4), alpha_filter))})')
    _G.add_data(temp_vdc1_list, [f'{titles[0]} (Filter = {s})' for s in alpha_filter], true_if_matrix=True, title_matrix=f'{titles[0]} (Filter = {", ".join(map(str, alpha_filter))})')
    _G.add_data(temp_vdc2, titles[1])
    _G.add_data(temp_vdc3, titles[2])
    # _G.add_data(temp_vdc_th_list1, [f'{titles[0][:titles[0].find("(")]} - {Test[1][Test[0]]} THs (Filter = {s})' for s in alpha_filter], true_if_matrix=True, title_matrix=f'{titles[0][:titles[0].find("(")]} (Filter = {", ".join(map(str, alpha_filter))})')
    _G.add_data(temp_vdc_th_list1, [f'{titles[0]} - {test_type[1][test_type[0]]} THs (Moving AVG = {s + 4})' for s in alpha_filter], true_if_matrix=True, title_matrix=f'{titles[0]} (Moving AVG = {", ".join(map(lambda x: str(x + 4), alpha_filter))})')
    _G.add_data(temp_vdc_th_list2, f'{titles[1]} - {test_type[1][test_type[0]]} THs')
    _G.add_data(temp_vdc_th_list3, f'{titles[2]} - {test_type[1][test_type[0]]} THs')
    # _G.set_slider([[n, n + 2 + len(alpha_filter)] for n in range(len(alpha_filter))], [f'Moving AVG = {s + 4}' for s in alpha_filter])
    _G.set_slider([[n, n + 2 + len(alpha_filter)] for n in range(len(alpha_filter))], [f'AVG = {s}' for s in alpha_filter])
    # _G.change_name(f'{file_name[:11]} plot {file_name[12:-4]} - Moving AVG')
    if Ks_not_float[0]:
        _G.change_name(f'{file_name[:10]} Ks plot {file_name[12:-4]} - {test_type[1][test_type[0]]}')
    else:
        _G.change_name(f'{file_name[:10]} plot {file_name[12:-4]} - {test_type[1][test_type[0]]}')
    _G.plot()
    _G.reset()

print()