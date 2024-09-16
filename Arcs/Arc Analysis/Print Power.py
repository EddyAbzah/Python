import os
import sys
import inspect
from datetime import datetime
import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import arc_th_calc
import log_file
import log_spi_or_scope
from my_pyplot import plot as _P, print_chrome as _PC, clear as _PP, print_lines as _PL, send_mail as _SM
import Plot_Graphs_with_Sliders as _G
import my_tools


# # Inverter stuff:
inverter_type = 'Jupiter_DSP'
log_with_time_stamp = True
# # Folders and filters:
path_logs = r'V:\HW_Infrastructure\Analog_Team\ARC_Data\Results\Jupiter\Jupiter+ Improved (7E0872F4-EC)\V-DC with double Grace Timer\Overpower Arcs 03 - full power (16-11-2022)'
path_output = path_logs + r'\Graphs'
path_logs_string_filter = 'rec'
# # Chrome action for plotly:
kill_chrome = False
auto_open_chrome_output = False
# # Energy Rise and Current Drop parameters:
window_size_1 = 20
filter_size_1 = 15
over_th_limit_1 = 12
# ## set sync_voltage_detection (in "arc_th_calc.py") to 9 if 5;5;3  _  or to 10 if 3;3;2
window_size_2 = 20
filter_size_2 = 15
over_th_limit_2 = 12


log_file.choose_inverter(inverter_string=inverter_type, with_time_stamp=log_with_time_stamp)
log_files_after_filter, log_file_names, log_arc_detections = list(log_file.get_files(folder_path=path_logs, string_filter=path_logs_string_filter))
_G.change_path(path_output)
_G.change_action(kill_chrome, auto_open_chrome_output, new_shared_x_axis=True, new_slider_prefix='Slider')
_G.change_action(new_slider_prefix='')

for index_file, file in enumerate(log_files_after_filter):
    print(f'Plotting record number {index_file} - {log_file_names[index_file]}:')

    all_logs = log_file.get_logs_all(file)
    energy_th_list = arc_th_calc.energy_rise_algo2(all_logs['log_energy'], window_size=window_size_1, filter_size=filter_size_1, over_th_limit=over_th_limit_1)
    current_th_list = arc_th_calc.current_drop_algo2(all_logs['log_current'], window_size=window_size_1, filter_size=filter_size_1, over_th_limit=over_th_limit_1)
    voltage_th_list = arc_th_calc.current_drop_algo2(all_logs['log_voltage'], window_size=window_size_1, filter_size=filter_size_1, over_th_limit=over_th_limit_1)
    # print(f'{index_file} = {max(current_th_list):0.02}')
    # continue
    _G.add_data(all_logs['log_energy'], 'log_energy')
    _G.add_data(energy_th_list, 'energy_th_list')
    _G.add_data(all_logs['log_energy_state'], 'log_energy_state')
    _G.add_data(all_logs['log_current'], 'log_current')
    _G.add_data(current_th_list, 'current_th_list')
    _G.add_data(all_logs['log_current_state'], 'log_current_state')
    _G.add_data(all_logs['log_voltage'], 'log_voltage')
    _G.add_data(voltage_th_list, 'voltage_th_list')
    _G.add_data(all_logs['log_voltage_state'], 'log_voltage_state')
    _G.add_data(all_logs['log_state'], 'log_state', -1)
    _G.set_slider([8, 9], ['voltage', 'state'])
    # _G.add_data([all_logs['log_energy_state'], all_logs['log_current_state']], ['log_energy_state, log_current_state'], true_if_matrix=True)



    _G.change_name(log_file_names[index_file])
    _G.plot()
    _G.reset()
