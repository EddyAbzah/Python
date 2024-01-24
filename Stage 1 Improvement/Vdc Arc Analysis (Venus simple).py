import pandas as pd
import arc_th_calc
import log_file
import glob
from my_pyplot import omit_plot as _O, plot as _P, print_chrome as _PC, clear as _PP, print_lines as _PL, send_mail as _SM
import Plot_Graphs_with_Sliders as _G
import my_tools


path_in = r'V:\HW_Infrastructure\Analog_Team\ARC_Data\Results\Jupiter\Jupiter+ Improved (7E0872F4-EC)\V-DC Bundled FW versions\Standard Arcs 01 - 3A only (02-01-2022)'
path_out = path_in + '\\Graphs'
file_filter = '*spi*.txt'
alpha_filter = [0.01, 0.005, 0.001, 0.0007, 0.0003, 0.0001, 0.00007, 0.00004]
sample_rate_spi = 16667
sample_rate_pwr = 35
cut_spi_samples = [False, 20000, None]   # Start, Stop
Vdc_Columns = [1, 2, 3]
Vdc_Columns = [1, 2, 3]
Vdc_Names = ['Vdc fast', 'Vdc Filtered (SPI case 630)', 'Vdc AVG Window']
Vdc_cut_if_above_below = [False, 752, 748]
plot_only_slow_vdc = False
plot_only_fast_vdc = [True, 5]   # index of the Filter
DEBUG = False

_G.change_path(path_out)
_G.change_action(new_kill_chrome=False, new_auto_open_chrome_output=True, new_shared_x_axis=True, new_slider_prefix='')

list_of_files = glob.glob(f'{path_in}\\{file_filter}')
for index_file, file in enumerate(list_of_files):
    temp_csv = pd.read_csv(file, skiprows=0)
    temp_vdc = temp_csv.iloc[:, Vdc_Columns[0]]
    if Vdc_cut_if_above_below[0]:
        print(f'len(temp_vdc) BEFORE = {len(temp_vdc)}')
        temp_vdc = temp_vdc[:next((i for i, x in enumerate(temp_vdc) if x > Vdc_cut_if_above_below[1] or x < Vdc_cut_if_above_below[2]), None)]
        print(f'len(mana) AFTER = {len(temp_vdc)}')
    record_time = len(temp_vdc) / sample_rate_spi

    if not plot_only_slow_vdc:
        temp_vdc1 = []
        temp_vdc_th_list1 = []
        for index_alpha, alpha in enumerate(alpha_filter):
            if not plot_only_fast_vdc[0] or index_alpha == plot_only_fast_vdc[1]:
                print(f'Vdc1 => Filtering the SPI')
                if cut_spi_samples[0]:
                    temp_vdc1.append(pd.Series(log_file.alpha_beta_filter(temp_vdc, alpha)[cut_spi_samples[1]:cut_spi_samples[2]]))
                else:
                    temp_vdc1.append(pd.Series(log_file.alpha_beta_filter(temp_vdc, alpha)))
                print(f'Vdc1 => Downsampling')
                temp_vdc1[-1] = pd.Series(log_file.avg_no_overlap(temp_vdc1[-1], sample_rate_spi, sample_rate_pwr))
                print(f'Vdc1 => Calculating the TH')
                temp_vdc_th_list1.append(arc_th_calc.voltage_drop_algo2(temp_vdc1[-1], window_size=20, filter_size=15, over_th_limit=12))
    if DEBUG:
        temp_vdc1 = temp_vdc1[0]
        temp_vdc_th_list1 = temp_vdc_th_list1[0]
    if not plot_only_fast_vdc[0]:
        temp_vdc = temp_csv.iloc[:, Vdc_Columns[1]]
        if Vdc_cut_if_above_below[0]:
            print(f'len(temp_vdc) BEFORE = {len(temp_vdc)}')
            temp_vdc = temp_vdc[:next((i for i, x in enumerate(temp_vdc) if x > Vdc_cut_if_above_below[1] or x < Vdc_cut_if_above_below[2]), None)]
            print(f'len(mana) AFTER = {len(temp_vdc)}')
        print(f'Vdc2 => Dropping consecutive duplicates')
        temp_vdc2 = temp_vdc.loc[temp_vdc.shift() != temp_vdc].reset_index(drop=True)
        print(f'Vdc2 => Downsampling method')
        record_freq = len(temp_vdc2) / record_time
        # temp_vdc2 = pd.Series(log_file.avg_no_overlap(temp_vdc2, record_freq, sample_rate_pwr))
        temp_vdc2 = pd.Series(log_file.avg_with_overlap(temp_vdc2, 4, 4, 0))
        print(f'Vdc2 => Calculating the TH')
        temp_vdc_th_list2 = arc_th_calc.voltage_drop_algo2(temp_vdc2, window_size=20, filter_size=15, over_th_limit=12)

        temp_vdc = temp_csv.iloc[:, Vdc_Columns[2]]
        if Vdc_cut_if_above_below[0]:
            print(f'len(temp_vdc) BEFORE = {len(temp_vdc)}')
            temp_vdc = temp_vdc[:next((i for i, x in enumerate(temp_vdc) if x > Vdc_cut_if_above_below[1] or x < Vdc_cut_if_above_below[2]), None)]
            print(f'len(mana) AFTER = {len(temp_vdc)}')
        print(f'Vdc3 => Dropping consecutive duplicates')
        temp_vdc3 = temp_vdc.loc[temp_vdc.shift() != temp_vdc]
        print(f'Vdc3 => Calculating the TH')
        temp_vdc_th_list3 = arc_th_calc.voltage_drop_algo2(temp_vdc3, window_size=20, filter_size=15, over_th_limit=12)

    if not (plot_only_slow_vdc or plot_only_fast_vdc[0]):
        _G.add_data(temp_vdc1, [f'{Vdc_Names[0]} (af = {s})' for s in alpha_filter], true_if_matrix=True, title_matrix=f'THs list Fast (af = {", ".join(str(s) for s in alpha_filter)})')
        _G.add_data(temp_vdc2, Vdc_Names[1])
        _G.add_data(temp_vdc3, Vdc_Names[2])
        _G.add_data(temp_vdc_th_list1, [f'{Vdc_Names[0]} THs (af = {s})' for s in alpha_filter], true_if_matrix=True, title_matrix=f'THs list Fast (af = {", ".join(str(s) for s in alpha_filter)})')
        _G.add_data(temp_vdc_th_list2, f'{Vdc_Names[1]} THs)')
        _G.add_data(temp_vdc_th_list3, f'{Vdc_Names[2]} THs)')
        _G.set_slider([[n, n + 2 + len(alpha_filter)] for n in range(len(alpha_filter))], [f'Alpha Filter = {s}' for s in alpha_filter])
        _G.change_name(f'Jupiter+I Vdc plots Rec{index_file + 1:03}')
        _G.plot()
        _G.reset()
    if plot_only_slow_vdc:
        _G.add_data(temp_vdc3, f'Rec{index_file + 1:03} Vdc Slow', index=0, title_matrix='Vdc Slow')
        _G.add_data(temp_vdc_th_list3, f'Rec{index_file + 1:03} THs list', index=1, title_matrix='Vdc Slow Thresholds list')
    if plot_only_fast_vdc[0]:
        _G.add_data(temp_vdc1[plot_only_fast_vdc[1]], f'Rec{index_file + 1:03} Vdc Fast', index=0, title_matrix='Vdc Fast')
        _G.add_data(temp_vdc_th_list1[plot_only_fast_vdc[1]], f'Rec{index_file + 1:03} THs list', index=1, title_matrix='Vdc Fast Thresholds list')
if plot_only_slow_vdc:
    _G.change_name(f'Jupiter+I Vdc plots - Vdc slow only')
if plot_only_fast_vdc[0]:
    _G.change_name(f'Jupiter+I Vdc plots - Vdc fast only for alpha filter = {alpha_filter[plot_only_fast_vdc[1]]}')
if plot_only_slow_vdc or plot_only_fast_vdc[0]:
    _G.set_slider([[n, n + len(list_of_files)] for n in range(len(list_of_files))], [f'Rec{n + 1:03}' for n in range(len(list_of_files))])
    _G.plot()
    _G.reset()
