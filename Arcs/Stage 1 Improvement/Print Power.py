import os
import sys
import inspect
from datetime import datetime
import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
import arc_th_calc
import log_file
import log_spi_or_scope
from my_pyplot import omit_plot as _O, plot as _P, print_chrome as _PC, clear as _PP, print_lines as _PL, send_mail as _SM
import Plot_Graphs_with_Sliders as _G
import my_tools


# # Inverter stuff:
inverter_type = 'jupiter_dsp'
log_with_time_stamp = True
skip_if_not_pwr = False
# # Folders and filters:
path_logs = r'M:\Users\HW Infrastructure\PLC team\ARC\Temp-Eddy\Test_Nonstandard_Arcs 22_02_23 14_42 6A ARC TEST4 with eddy version v4'
path_output = path_logs + r'\Graphs'
path_logs_string_filter = 'pwr'
# # Chrome action for plotly:
kill_chrome = False
auto_open_chrome_output = False
cut_logs = [False, 0, [877, 877, 860, 832, 851, 819]]   # Start, Finish
# # Energy Rise and Current Drop parameters:
window_size_1 = 20
filter_size_1 = 15
over_th_limit_1 = 12
# ## set sync_voltage_detection (in "arc_th_calc.py") to 9 if 5;5;3  _  or to 10 if 3;3;2
window_size_2 = 20
filter_size_2 = 15
over_th_limit_2 = 12


log_file.choose_inverter(inverter_string=inverter_type, with_time_stamp=log_with_time_stamp)
log_files_after_filter, log_file_names, log_arc_detections = list(log_file.get_files(folder_path=path_logs, string_filter=path_logs_string_filter, skip_if_not_pwr=skip_if_not_pwr))
_G.change_path(path_output)
_G.change_action(kill_chrome, auto_open_chrome_output, new_shared_x_axis=True, new_slider_prefix='')
if type(cut_logs[2]) is list:
    index_cut = 0

for index_file, file in enumerate(log_files_after_filter):
    print(f'Plotting record number {index_file} - {log_file_names[index_file]}:')

    all_logs = log_file.get_logs_all(file)
    if cut_logs[0]:
        if type(cut_logs[2]) is list:
            for key in all_logs.keys():
                all_logs.update({key: all_logs[key][cut_logs[1]:cut_logs[2][index_cut]]})
            index_cut += 1
        else:
            for key in all_logs.keys():
                all_logs.update({key: all_logs[key][cut_logs[1]:cut_logs[2]]})

    energy_th_list = arc_th_calc.energy_rise_algo2(all_logs['log_energy'], window_size=window_size_1, filter_size=filter_size_1, over_th_limit=over_th_limit_1)
    current_th_list = arc_th_calc.current_drop_algo2(all_logs['log_current'], window_size=window_size_1, filter_size=filter_size_1, over_th_limit=over_th_limit_1)
    voltage_th_list = arc_th_calc.current_drop_algo2(all_logs['log_voltage'], window_size=window_size_1, filter_size=filter_size_1, over_th_limit=over_th_limit_1)

    # _P(all_logs['log_energy'])
    # _P(all_logs['log_energy_state'])
    # _P(arc_th_calc.energy_rise_algo_old(all_logs['log_energy'], 8))
    # _P(arc_th_calc.energy_rise_floor(all_logs['log_energy']))
    # _P(all_logs['log_current'])
    # _P(arc_th_calc.current_drop_algo2(all_logs['log_current']))
    # _P(all_logs['log_voltage'])
    # _P(arc_th_calc.voltage_drop_algo2(all_logs['log_voltage']))

    # _G.add_data(arc_th_calc.voltage_drop_algo2(all_logs['log_voltage']), "Voltage slow interrupt (Buffer)")
    # _G.add_data(arc_th_calc.voltage_drop_algo2(all_logs['log_voltage2']), "ADC_GetVdcP() - ADC_GetVdcM() = medium")
    # _G.add_data(arc_th_calc.voltage_drop_algo2(all_logs['log_voltage1']), "ADC_GetVdc() = fast")
    # continue


    for mana in range(2):
        if mana == 0:
            delete_indexes = None
        else:
            delete_indexes = all_logs['log_state']
        _G.add_data(all_logs['log_energy'], 'log_energy', delete_indexes=delete_indexes)
        _G.add_data(energy_th_list, 'energy_th_list', delete_indexes=delete_indexes)
        _G.add_data(all_logs['log_energy_state'], 'log_energy_state', delete_indexes=delete_indexes)
        _G.add_data(all_logs['log_current'], 'log_current', delete_indexes=delete_indexes)
        _G.add_data(current_th_list, 'current_th_list', delete_indexes=delete_indexes)
        _G.add_data(all_logs['log_current_state'], 'log_current_state', delete_indexes=delete_indexes)
        _G.add_data(all_logs['log_voltage'], 'log_voltage', delete_indexes=delete_indexes)
        _G.add_data(voltage_th_list, 'voltage_th_list', delete_indexes=delete_indexes)
        _G.add_data(all_logs['log_voltage_state'], 'log_voltage_state', delete_indexes=delete_indexes)
        # _G.add_data(all_logs['log_state'], 'log_state', -1)
        # _G.set_slider([8, 9], ['voltage', 'state'])
        # _G.add_data([all_logs['log_energy_state'], all_logs['log_current_state']], ['log_energy_state, log_current_state'], true_if_matrix=True)

        if mana == 0:
            _G.change_name(log_file_names[index_file])
        else:
            _G.change_name(log_file_names[index_file][:log_file_names[index_file].find('pwr')] + 'single state ' + log_file_names[index_file][log_file_names[index_file].find('pwr'):])
        _G.plot()
        _G.reset()
    continue
