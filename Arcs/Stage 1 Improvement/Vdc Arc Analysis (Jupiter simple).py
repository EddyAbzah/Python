import pandas as pd
import arc_th_calc
import log_file
from my_pyplot import omit_plot as _O, plot as _P, print_chrome as _PC, clear as _PP, print_lines as _PL, send_mail as _SM
import Plot_Graphs_with_Sliders as _G
import my_tools


path_in = r'V:\HW_Infrastructure\Analog_Team\ARC_Data\Results\Venus3\11.4kW SEDSP with FHB 74861BDA\Vdc algorithm with BUI\Vdc Arcs 01 (11-12-2022)'
path_out = path_in # + '\\Graphs'
alpha_filter = [0.001, 0.0005, 0.0001]
sample_rate_spi = 16667
sample_rate_vdc = 100
sample_rate_pwr = 35

_G.change_path(path_out)
_G.change_action(new_kill_chrome=False, new_auto_open_chrome_output=True, new_shared_x_axis=True, new_slider_prefix='')

for i in range(1, 6):
    temp_csv = pd.read_csv(path_in + f'\\Jupiter spi Rec00{i}.txt', skiprows=2)
    temp_vdc1 = temp_csv.iloc[:, 1]
    temp_vdc_th_list1 = pd.Series(arc_th_calc.voltage_drop_algo2(temp_vdc1, window_size=20, filter_size=15, over_th_limit=12))

    temp_csv = pd.read_csv(path_in + f'\\Jupiter spi Rec00{i}.txt', skiprows=1)
    temp_vdc2 = temp_csv.iloc[:, 0] * 0.0625
    temp_vdc3 = temp_csv[temp_csv.iloc[:, 1] > 0].iloc[:, 1] * 0.0625

    temp_vdc2_list = []
    temp_vdc_th_list2 = []

    for alpha in alpha_filter:
        print(f'Vdc2 => Filtering the SPI')
        temp_vdc2_list.append(pd.Series(log_file.alpha_beta_filter(temp_vdc2, alpha)))
        print(f'Vdc2 => Downsampling')
        temp_vdc2_list[-1] = pd.Series(log_file.avg_no_overlap(temp_vdc2_list[-1], sample_rate_spi, sample_rate_pwr))
        print(f'Vdc2 => Calculating the TH')
        temp_vdc_th_list2.append(arc_th_calc.voltage_drop_algo2(temp_vdc2_list[-1], window_size=20, filter_size=15, over_th_limit=12))

    print(f'Vdc3 => Dropping consecutive duplicates')
    temp_vdc3 = temp_vdc3.loc[temp_vdc3.shift() != temp_vdc3]
    print(f'Vdc3 => Downsampling method')
    temp_vdc3 = pd.Series(log_file.avg_no_overlap(temp_vdc3, sample_rate_vdc, sample_rate_pwr))
    print(f'Vdc3 => Calculating the TH')
    temp_vdc_th_list3 = arc_th_calc.voltage_drop_algo2(temp_vdc3, window_size=20, filter_size=15, over_th_limit=12)

    _G.add_data(temp_vdc1, 'Vdc PWR log')
    _G.add_data(temp_vdc2_list, [f'Vdc fast (af = {s})' for s in alpha_filter], true_if_matrix=True)
    _G.add_data(temp_vdc3, 'Vdc Slow')
    _G.add_data(temp_vdc_th_list1, 'temp_vdc_th_list1')
    _G.add_data(temp_vdc_th_list2, [f'temp_vdc_th_list2 (af = {s})' for s in alpha_filter], true_if_matrix=True)
    _G.add_data(temp_vdc_th_list3, 'temp_vdc_th_list3')
    # _G.set_slider([6, 7, 8], [f'Alpha Filter = {s}' for s in alpha_filter])
    _G.change_name(f'Jupiter plots Rec00{i}')
    _G.plot()
    _G.reset()

print()