import os
import gc
import sys
import inspect
from datetime import datetime
# import math
import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import arc_th_calc
import log_file
import log_spi_or_scope
from my_pyplot import omit_plot as _O, plot as _P, print_chrome as _PC, clear as _PP, print_lines as _PL, send_mail as _SM
import Plot_Graphs_with_Sliders as _G
import my_tools


# # txt output instead of the console - ATTENTION - if True, there will be no Console output:
output_text = False
path_txt = datetime.now().strftime("%d-%m-%Y %H-%M-%S")
path_txt = f'Terminal Log ({path_txt}).txt'
# # Chrome action for plotly:
plot_offline = False
kill_chrome = False and plot_offline
auto_open_chrome_output = False and plot_offline
# # Inverter stuff:
inverter_type = 'Jupiter_DSP'
log_with_time_stamp = True
# # Test type:
# # 1 = Regular Arc search (via State Machine).
# # 2 = False Alarms (no Arc search).
# # 3 = False Alarms, BUT, records will be cut manually when the state machine is = 10.
# # 4 = No Filter.
test_type = 4
# # divide the number of log per plot (in other words, records in each html output):
plots_per_test = 1
# # Folders and filters:
path_output = r'C:\Users\eddy.a\Downloads'
path_logs = r'C:\Users\eddy.a\Downloads'
path_logs_string_filter = 'rec'
# # this will run the voltage algorithm on SPI records instead of the PWR
take_spi_voltage = True
spi_log_column = 'VDC'
# ## Venus3 DSP: ## sample_rate_spi = 50e3
sample_rate_spi = 16667
sample_rate_pwr = 28.6
sync_pwr_to_spi_1 = (True, 0)
sync_pwr_to_spi_2 = 4600
# # plot name ('.html' ass added later):
plot_name = f'Spread Spectrum Test'
# # divide the output plots into 2D figures:
plot_columns = 3
# # Energy Rise and Current Drop parameters:
energy_rise_th_steps = np.array([12, 10, 8, 6])    # Best = np.arange(6, 17, 1)
current_drop_th_steps = np.array([0.2, 0.15, 0.1])      # Best = np.array([0.2, 0.15, 0.1])
window_size_1 = 20
filter_size_1 = 15
over_th_limit_1 = 12
# # Voltage Drop parameters:
voltage_drop_th_steps = np.arange(0, 0.05, 0.0025)
# ## set sync_voltage_detection (in "arc_th_calc.py") to 9 if 5;5;3  _  or to 10 if 3;3;2
window_size_2 = 15
filter_size_2 = 15
over_th_limit_2 = 12
voltage_type_1 = f'W={window_size_2}; F={filter_size_2}; T={over_th_limit_2}'
# # Moving average calculations:
Noise_floor_AVG_samples_1 = 50
Noise_floor_AVG_samples_2 = 100
# # Enable to print MIN, MAX and AVG of each DF:
test_type_print_ranges = False
# # Add Data Callouts (labels) to the plots:
add_data_labels = False
# # Find Arc starts using "np.diff()" = very inconsistent
find_arc_start = False
# # Test types - not both True - either one = True or False:
test_type_dc_voltage_1 = False
test_type_dc_voltage_2 = False
test_type_find_arc_start_only = False and find_arc_start
test_type_count_false_alarms = True
# # If all of the above are False:
print_voltage_and_power = False
# # Alpha / Beta Filter = smaller alpha, higher filtration:
if test_type_dc_voltage_1:
    alpha_filter = [a / 100 for a in range(100, 0, -10)]
else:
    alpha_filter = 1
    voltage_T_avg_F_min = False
    if voltage_T_avg_F_min:
        voltage_type_2 = 'AVG'
    else:
        voltage_type_2 = 'MIN'


def main():
    import pandas as pd
    global alpha_filter
    alpha_filter = 0.003
    true_if_avg = True
    spi_log_voltage1 = pd.read_csv(r"C:\Users\eddy.a\Downloads\20__20 (100 and 0) -  Arc in 100 rec1.csv")
    spi_log_voltage2 = log_file.convert_to_df(log_file.alpha_beta_filter(spi_log_voltage1['VDC'], alpha_filter))
    spi_log_voltage3 = log_file.convert_to_df(log_file.avg_no_overlap(spi_log_voltage2, sample_rate_spi, 35))
    voltage_th_list = arc_th_calc.voltage_algo(log_file.convert_to_df(spi_log_voltage3), window_size=20, filter_size=15, over_th_limit=12, true_if_avg=true_if_avg)
    exit()



    # if output_text and inspect.currentframe().f_code.co_name == 'main':
    #     default_stdout = sys.stdout
    #     if not os.path.exists(path_output):
    #         os.makedirs(path_output)
    #     sys.stdout = open(f'{path_output}/{path_txt}', 'w')
    # global voltage_T_avg_F_min
    # global voltage_type_2
    # global alpha_filter
    # for T_avg_F_min in [False, True]:
    #     for alpha in [0.003, 0.001, 0.0005, 0.0001, 0.00005]:
    #         alpha_filter = alpha
    #         voltage_T_avg_F_min = T_avg_F_min
    #         if voltage_T_avg_F_min:
    #             voltage_type_2 = 'AVG'
    #         else:
    #             voltage_type_2 = 'MIN'
    #         print()
    #         print()
    #         print(f'voltage_T_avg_F_min changed to = {T_avg_F_min} ({voltage_type_2})')
    #         print(f'alpha_filter changed to = {alpha}')
    #         print()
    #         print()
    #         run_main()
    #         gc.collect()
    # if output_text and inspect.currentframe().f_code.co_name == 'main':
    #     sys.stdout.close()
    #     sys.stdout = default_stdout


def run_main():
    global sync_pwr_to_spi_2
    list_of_figs = []
    if find_arc_start:
        list_find_arc_start = []
    if test_type_find_arc_start_only:
        list_energy_df = []
        list_current_df = []
        list_voltage_df = []
    if test_type_count_false_alarms:
        list_false_alarms_count = []
    if kill_chrome:
        os.system("taskkill /im chrome.exe /f")
    if output_text and inspect.currentframe().f_code.co_name == 'main':
        default_stdout = sys.stdout
        if not os.path.exists(path_output):
            os.makedirs(path_output)
        sys.stdout = open(f'{path_output}/{path_txt}', 'w')
    print(f"Starting Python... Time: {str(datetime.now())[:-7]}")
    log_file.choose_inverter(inverter_string=inverter_type, with_time_stamp=log_with_time_stamp)
    log_files_after_filter, log_file_names, log_arc_detections = list(
        log_file.get_files(folder_path=path_logs, string_filter=path_logs_string_filter))
    if test_type_dc_voltage_1:
        plot_titles = ['Energy [dB]', 'DC Voltage [V]', 'DC Voltage * Current Drop [V/I]',
                       'AC Current [A]', 'MIN drop algorithm 1', 'MIN drop algorithm 2',
                       'Current Drop [A]', 'AVG drop algorithm 1', 'AVG drop algorithm 2']
    elif test_type_dc_voltage_2:
        plot_titles = ['Energy [dB]', 'Energy Rise [dB]', 'Detection - Energy Rise and Current Drop together',
                       'DC Voltage [V]', f'Voltage Drop [V] {voltage_type_2} for {voltage_type_1}', 'Detection - Voltage Drop only',
                       'AC Current [A]', 'Current Drop [A]', 'Detection - all three Detections together']
    else:
        if print_voltage_and_power:
            plot_titles = ['Energy [dB]', 'AC Current [A]', 'DC Voltage [V]', 'Energy Rise [dB]', 'Current Drop [A]',
                           'Power [W]', 'Detection - Energy Rise', 'Detection - Current Drop', 'Arc Detection']
        else:
            # # with moving average of the noise floor:
            plot_titles = ['Energy [dB]', 'AC Current [A]', f'Noise floor AVG ({Noise_floor_AVG_samples_1} samples)',
                           'Energy Rise [dB]', 'Current Drop [A]', f'Noise floor AVG ({Noise_floor_AVG_samples_2} samples)',
                           'Detection - Energy Rise', 'Detection - Current Drop', 'Arc Detection']

    plot_rows = int(len(plot_titles) / plot_columns)
    all_specs = np.reshape([[{"secondary_y": False}] for x in range(len(plot_titles))],
                           (plot_rows, plot_columns)).tolist()
    for index_file, file in enumerate(log_files_after_filter):
        if plots_per_test == 1:
            string_index_file = ''
        else:
            string_index_file = f'{log_file_names[index_file][:6]} - '
        print(f'Plotting record number {string_index_file}- {log_file_names[index_file]}:')

        (log_energy, log_current, log_voltage, log_power, log_state, log_energy_before_stage2, log_current_before_stage2,
         log_voltage_before_stage2, log_power_before_stage2, cut_log_at) = log_file.get_logs(file, test_type=test_type, extra_prints=8)
        if find_arc_start:
            list_find_arc_start.append(log_file.find_arc_via_dc_voltage(log_file_names[index_file][:6], log_voltage))
        if test_type_find_arc_start_only:
            list_energy_df.append(log_energy)
            list_current_df.append(log_current)
            list_voltage_df.append(log_voltage)
            continue
        energy_th_list = arc_th_calc.plot_all(log_energy_before_stage2, window_size=window_size_1,
                                              filter_size=filter_size_1, over_th_limit=over_th_limit_1)
        current_th_list = arc_th_calc.current_algo_old(log_current_before_stage2, window_size=window_size_1,
                                                       filter_size=filter_size_1, over_th_limit=over_th_limit_1)
        if test_type_dc_voltage_2:
            list_voltage_alpha_filtered = log_file.convert_to_df(log_file.alpha_beta_filter(df=log_voltage, alpha=alpha_filter))
            voltage_th_list = arc_th_calc.voltage_algo(list_voltage_alpha_filtered,
                                                       window_size=window_size_2, filter_size=filter_size_2,
                                                       over_th_limit=over_th_limit_2, true_if_avg=voltage_T_avg_F_min)
        if test_type_dc_voltage_1:
            list_voltage_alpha_filtered = []
            list_voltage_with_current = []
            voltage_th_list_min_1 = []
            voltage_th_list_min_2 = []
            voltage_th_list_avg_1 = []
            voltage_th_list_avg_2 = []
            for step_alpha in alpha_filter:
                list_voltage_alpha_filtered.append(log_file.convert_to_df(
                    log_file.alpha_beta_filter(df=log_voltage, alpha=step_alpha)))
                list_voltage_with_current.append(log_file.convert_to_df(
                    [a * b for a, b in zip(list_voltage_alpha_filtered[-1], current_th_list)]))
            print(f'Getting the Voltage Drop')
            for step_alpha in range(len(alpha_filter)):
                voltage_th_list_min_1.append(arc_th_calc.voltage_algo(list_voltage_alpha_filtered[step_alpha],
                                                                    window_size=window_size_2, filter_size=filter_size_2,
                                                                    over_th_limit=over_th_limit_2, true_if_avg=False))
                voltage_th_list_avg_1.append(arc_th_calc.voltage_algo(list_voltage_alpha_filtered[step_alpha],
                                                                    window_size=window_size_2, filter_size=filter_size_2,
                                                                    over_th_limit=over_th_limit_2, true_if_avg=True))
                voltage_th_list_min_2.append(arc_th_calc.voltage_algo(list_voltage_with_current[step_alpha],
                                                                      window_size=window_size_2,
                                                                      filter_size=filter_size_2,
                                                                      over_th_limit=over_th_limit_2, true_if_avg=False))
                voltage_th_list_avg_2.append(arc_th_calc.voltage_algo(list_voltage_with_current[step_alpha],
                                                                      window_size=window_size_2,
                                                                      filter_size=filter_size_2,
                                                                      over_th_limit=over_th_limit_2, true_if_avg=True))
        else:
            if not test_type_dc_voltage_2 and not print_voltage_and_power:
                # # with moving average of the noise floor:
                noise_floor_moving_avg_list_1 = arc_th_calc.noise_floor_AVG(log_energy, Noise_floor_AVG_samples_1)
                noise_floor_moving_avg_list_2 = arc_th_calc.noise_floor_AVG(log_energy, Noise_floor_AVG_samples_2)
            if test_type_count_false_alarms:
                if take_spi_voltage:
                    log_spi_files, log_spi_names, log_scope_files, log_scope_names = list(log_spi_or_scope.get_files(
                        folder_path=path_logs, string_filter=path_logs_string_filter, spi_log_column=spi_log_column,
                        file_name=log_file_names[index_file][:-4].lower().replace("pwr", "spi")))
                    spi_log_voltage = log_file.convert_to_df(log_file.alpha_beta_filter(log_spi_files[0], alpha_filter))
                    if sync_pwr_to_spi_1[0]:
                        list_voltage_alpha_filtered = log_file.convert_to_df(log_file.avg_no_overlap(
                            spi_log_voltage[sync_pwr_to_spi_2:], sample_rate_spi, sample_rate_pwr))
                        if sync_pwr_to_spi_1[1] < 0:
                            list_voltage_alpha_filtered = log_file.convert_to_df(
                                [0] * abs(sync_pwr_to_spi_1[1]) + list(list_voltage_alpha_filtered))
                        else:
                            list_voltage_alpha_filtered = list_voltage_alpha_filtered[sync_pwr_to_spi_1[1]:]
                        max_correlation = 0
                    else:
                        for sample_rate_new in np.arange(17.3, 22, 0.1):
                            list_voltage_alpha_filtered = log_file.convert_to_df(log_file.avg_no_overlap(
                                spi_log_voltage[sync_pwr_to_spi_2:], sample_rate_spi, sample_rate_new))
                            max_correlation = log_file.get_max_correlation(log_voltage, list_voltage_alpha_filtered)
                            print(f'{log_file_names[index_file]}: max_correlation = {max_correlation}')
                            if max_correlation > 100:
                                sync_pwr_to_spi_2 = sync_pwr_to_spi_2 + 5000
                                print(f'{log_file_names[index_file]}: new sync_pwr_to_spi_2 = {sync_pwr_to_spi_2}')
                                continue
                            if max_correlation == 0 or max_correlation < 0:
                                break
                    if max_correlation > 50:
                        print(f'ERRRRRRRRRRRRRRROR::: skipping {log_file_names[index_file]}... BYE!')
                        print(f'{log_file_names[index_file]}: max_correlation = {max_correlation}')
                        continue
                else:
                    list_voltage_alpha_filtered = log_file.alpha_beta_filter(df=log_voltage, alpha=alpha_filter)
                voltage_th_list = arc_th_calc.voltage_algo(log_file.convert_to_df(list_voltage_alpha_filtered),
                                                           window_size=window_size_2, filter_size=filter_size_2,
                                                           over_th_limit=over_th_limit_2, true_if_avg=voltage_T_avg_F_min)
            list_detection_energy = []
            list_detection_current = []
            list_detection_inverter = []
            for step_current in current_drop_th_steps:
                list_detection_current.append(current_th_list >= step_current)
            for index_energy, step_energy in enumerate(energy_rise_th_steps):
                list_detection_energy.append(energy_th_list >= step_energy)
                for index_current, step_current in enumerate(current_drop_th_steps):
                    list_detection_inverter.append([a and b for a, b in zip(list_detection_energy[index_energy], list_detection_current[index_current])])
            if test_type_dc_voltage_2 or test_type_count_false_alarms:
                list_detection_voltage = []
                list_detection_inverter_with_voltage = []
                for step_voltage in voltage_drop_th_steps:
                    list_detection_voltage.append(voltage_th_list >= step_voltage)
                for index_energy, step_energy in enumerate(energy_rise_th_steps):
                    for index_current, step_current in enumerate(current_drop_th_steps):
                        for index_voltage, step_voltage in enumerate(voltage_drop_th_steps):
                            list_detection_inverter_with_voltage.append(log_file.combine_detections(
                                [a and b for a, b in zip(list_detection_energy[index_energy], list_detection_current[index_current])],
                                list_detection_voltage[index_voltage]))
                if test_type_count_false_alarms:
                    index_1 = 0
                    index_2 = 0
                    list_false_alarms = []
                    for step_energy in energy_rise_th_steps:
                        for step_current in current_drop_th_steps:
                            diff = np.diff(np.multiply(list_detection_inverter[index_1], 1))
                            detection_indexes = np.argwhere(diff == 1).flatten()
                            list_false_alarms.append({"Detection": 'Energy + Current', "Energy TH": step_energy,
                                                      "Current TH": f'{step_current:.2}', "Voltage TH": 'disabled',
                                                      "Count": len(detection_indexes)})
                            index_1 += 1
                            for step_voltage in voltage_drop_th_steps:
                                if step_voltage == 0:
                                    index_2 += 1
                                    continue
                                diff = np.diff(np.multiply(list_detection_inverter_with_voltage[index_2], 1))
                                detection_indexes = np.argwhere(diff == 1).flatten()
                                list_false_alarms.append({"Detection": 'All', "Energy TH": step_energy,
                                                          "Current TH": f'{step_current:.2}', "Voltage TH": f'{step_voltage:.2}',
                                                          "Count": len(detection_indexes)})
                                index_2 += 1
                    list_false_alarms_count.append(list_false_alarms)
                    continue
        if test_type_print_ranges:
            if test_type_dc_voltage_1:
                arc_th_calc.print_ranges_2(index_file=index_file + 1, file_name=log_file_names[index_file],
                                           print_list=[voltage_th_list_min_1, voltage_th_list_avg_1,
                                                       voltage_th_list_min_2, voltage_th_list_avg_2],
                                           print_titles=['MIN drop algorithm 1', 'AVG drop algorithm 1',
                                                         'MIN drop algorithm 2', 'AVG drop algorithm 2'],
                                           alpha_filter=alpha_filter)
            else:
                # # For printing MIN, MAX and AVG of each DF:
                arc_th_calc.print_ranges(index_file=index_file + 1, print_list=[log_energy, energy_th_list, current_th_list],
                                         file_name=log_file_names[index_file], cut_log_at=cut_log_at, find_peaks=True,
                                         print_titles=['Energy [dB]', f'Energy Rise (Window Size {window_size_1})',
                                                       f'Current TH (Window Size {window_size_1})'])
        if test_type_dc_voltage_1:
            plot_list = [log_energy, list_voltage_alpha_filtered, list_voltage_with_current,
                         log_current, voltage_th_list_min_1, voltage_th_list_min_2,
                         current_th_list, voltage_th_list_avg_1, voltage_th_list_avg_2]
        elif test_type_dc_voltage_2:
            plot_list = [log_energy, energy_th_list, list_detection_inverter,
                         log_voltage, voltage_th_list, list_detection_voltage,
                         log_current, current_th_list, list_detection_inverter_with_voltage]
        else:
            if print_voltage_and_power:
                plot_list = [log_energy, log_current, log_voltage, energy_th_list, current_th_list, log_power_before_stage2,
                             list_detection_energy, list_detection_current, list_detection_inverter]
            else:
                # # with moving average of the noise floor:
                plot_list = [log_energy, log_current, noise_floor_moving_avg_list_1, energy_th_list, current_th_list,
                             noise_floor_moving_avg_list_2, list_detection_energy, list_detection_current, list_detection_inverter]

        if index_file % plots_per_test == 0:
            fig = make_subplots(subplot_titles=plot_titles, rows=plot_rows, cols=plot_columns,
                                specs=all_specs, shared_xaxes=True)
        if test_type_dc_voltage_1:
            fig_set_visible = index_file % plots_per_test * (3 + 6 * len(alpha_filter))
            for index, plot in enumerate(plot_titles):
                if index == 0:
                    fig_set_visible_delta = 0
                elif index % 3 == 0:
                    fig_set_visible_delta = fig_set_visible_delta + 1
                if index % 3 > 0:
                    for index_2, plot_2 in enumerate(plot_list[index]):
                        if index / 3 < 1:
                            trace_name = f"{string_index_file}DC Voltage - Alpha filter = {alpha_filter[index_2]}"
                        elif index / 3 < 2:
                            trace_name = f"{string_index_file}MIN Detection for filter = {alpha_filter[index_2]}"
                        else:
                            trace_name = f"{string_index_file}AVG Detection for filter = {alpha_filter[index_2]}"
                        fig.add_trace(go.Scatter(y=plot_2, name=trace_name, # legendgroup=f"group{index_file + 1}",
                                                 showlegend=True, visible=False), col=index % plot_columns + 1,
                                      row=int(index / plot_columns) + 1)
                    fig.data[fig_set_visible + fig_set_visible_delta + 1].visible = True
                    fig_set_visible_delta = fig_set_visible_delta + len(alpha_filter)
                else:
                    fig.add_trace(go.Scatter(y=plot_list[index], name=f"{string_index_file}{plot}"),
                                  col=index % plot_columns + 1, row=int(index / plot_columns) + 1)
        elif test_type_dc_voltage_2:
            fig_set_visible_delta_1 = len(current_drop_th_steps) * len(energy_rise_th_steps)
            fig_set_visible_delta_2 = fig_set_visible_delta_1 * len(voltage_drop_th_steps) + len(voltage_drop_th_steps)
            fig_set_visible = index_file % plots_per_test * (6 + fig_set_visible_delta_1 + fig_set_visible_delta_2)
            fig_set_visible_delta = len(list_detection_inverter) + len(list_detection_inverter_with_voltage) + len(list_detection_voltage)
            for index, plot in enumerate(plot_titles):
                if (index + 1) % 3 != 0:
                    fig.add_trace(go.Scatter(y=plot_list[index],  # legendgroup=f"group{index_file + 1}",
                                             name=f"{string_index_file}{plot}"),
                                  col=index % plot_columns + 1, row=int(index / plot_columns) + 1)
                elif index == 2:   # plotting Detection list for Energy Rise and Current Drop
                    for index_energy, step_energy in enumerate(energy_rise_th_steps):
                        for index_current, step_current in enumerate(current_drop_th_steps):
                            fig.add_trace(go.Scatter(# legendgroup=f"group{index_file + 1}",
                                                     y=plot_list[index][index_energy * len(current_drop_th_steps) + index_current],
                                                     name=f"{string_index_file}Detection for TH = {step_energy}[dB] @ {step_current:.2f}[A]",
                                                     showlegend=True, visible=False),
                                          col=index % plot_columns + 1, row=int(index / plot_columns) + 1)
                    fig.data[fig_set_visible + index].visible = True
                elif index == 5:   # plotting Detection list for Voltage Drop
                    for index_voltage, step_voltage in enumerate(voltage_drop_th_steps):
                        fig.add_trace(go.Scatter(y=plot_list[index][index_voltage],  # legendgroup=f"group{index_file + 1}",
                                                 name=f"{string_index_file}Detection for Voltage Drop TH = {step_voltage:.2f}[V]",
                                                 showlegend=True, visible=False), col=index % plot_columns + 1,
                                      row=int(index / plot_columns) + 1)
                    fig.data[fig_set_visible + index + len(list_detection_inverter) - 1].visible = True
                elif index == 8:   # plotting Detection list for all parameters
                    index_2 = 0
                    for index_energy, step_energy in enumerate(energy_rise_th_steps):
                        for index_current, step_current in enumerate(current_drop_th_steps):
                            for index_voltage, step_voltage in enumerate(voltage_drop_th_steps):
                                fig.add_trace(go.Scatter(y=plot_list[index][index_2],  # legendgroup=f"group{index_file + 1}",
                                                         name=f"{string_index_file}Detection for TH = {step_energy}[dB] @ {step_current:.2f}[A] @ {step_voltage:.2f}[V]",
                                                         showlegend=True, visible=False),
                                              col=index % plot_columns + 1, row=int(index / plot_columns) + 1)
                                index_2 += 1
                    fig.data[fig_set_visible + index + len(list_detection_inverter) + len(list_detection_voltage) - 2].visible = True
        else:
            fig_set_visible_delta_1 = len(current_drop_th_steps) + len(energy_rise_th_steps)
            fig_set_visible_delta_2 = len(current_drop_th_steps) * len(energy_rise_th_steps)
            fig_set_visible = index_file % plots_per_test * (6 + fig_set_visible_delta_2)
            for index, plot in enumerate(plot_titles):
                if index < 6:
                    fig.add_trace(go.Scatter(y=plot_list[index], # legendgroup=f"group{index_file + 1}",
                                             name=f"{string_index_file}{plot}"),
                                  col=index % plot_columns + 1, row=int(index / plot_columns) + 1)
                elif index == 6:   # plotting Detection list for Energy Rise
                    for index_2, plot_2 in enumerate(plot_list[index]):
                        fig.add_trace(go.Scatter(y=plot_2, # legendgroup=f"group{index_file + 1}",
                                                 name=f"{string_index_file}Detection for Energy Rise TH = {energy_rise_th_steps[index_2]}[dB]",
                                                 showlegend=True, visible=False), col=index % plot_columns + 1,
                                      row=int(index / plot_columns) + 1)
                    fig.data[fig_set_visible + index].visible = True
                elif index == 7:   # plotting Detection list for Current Drop
                    for index_2, plot_2 in enumerate(plot_list[index]):
                        fig.add_trace(go.Scatter(y=plot_2, # legendgroup=f"group{index_file + 1}",
                                                 name=f"{string_index_file}Detection for Current Drop TH = {current_drop_th_steps[index_2]:.2f}[A]",
                                                 showlegend=True, visible=False), col=index % plot_columns + 1,
                                      row=int(index / plot_columns) + 1)
                    fig.data[fig_set_visible + index + len(energy_rise_th_steps) - 1].visible = True
                else:   # plotting Detection list for Energy Rise AND Current Drop
                    for index_energy, step_energy in enumerate(energy_rise_th_steps):
                        for index_current, step_current in enumerate(current_drop_th_steps):
                            fig.add_trace(go.Scatter(# legendgroup=f"group{index_file + 1}",
                                                     y=plot_list[index][index_energy * (len(energy_rise_th_steps) - 1) + index_current],
                                                     name=f"{string_index_file}Detection for TH = {step_energy}[dB] @ {step_current:.2f}[A]",
                                                     showlegend=True, visible=False),
                                          col=index % plot_columns + 1, row=int(index / plot_columns) + 1)
                    fig.data[fig_set_visible + index + fig_set_visible_delta_1 + len(voltage_drop_th_steps) - 2].visible = True
        if index_file % plots_per_test == 0:
            list_of_figs.append(fig)
        print('------------------------------------------------------------------------------------------------------')
        print()
    if test_type_count_false_alarms:
        print('Record, Detection method, Energy TH, Current TH, Voltage TH, False Alarms Count')
        for index, list_false_alarms in enumerate(list_false_alarms_count):
            for false_alarms in list_false_alarms:
                print(log_file_names[index][:-4] + ', {Detection}, {Energy TH}, {Current TH}, {Voltage TH}, {Count}'.format(**false_alarms))
        print('All records together:')
        print('Detection method, Energy TH, Current TH, Voltage TH, False Alarms Count')
        list_of_counts = dict()
        for index, list_false_alarms in enumerate(list_false_alarms_count):
            for false_alarms in list_false_alarms:
                temp_string = ', '.join([str(a) for a in false_alarms.values()][:-1])
                if index == 0:
                    list_of_counts[temp_string] = false_alarms["Count"]
                else:
                    list_of_counts[temp_string] += false_alarms["Count"]
        for key, value in list_of_counts.items():
            print(f'{key}, {value}')
    elif plot_offline:
        if test_type_find_arc_start_only:
            slider_steps = []
            fig = make_subplots(subplot_titles=['Energy', 'Current', 'Voltage'], rows=3, cols=1, shared_xaxes=True)
            for df_index, (df_e, df_c, df_v) in enumerate(zip(list_energy_df, list_current_df, list_voltage_df)):
                if df_index == 0:
                    visible = True
                else:
                    visible = False
                fig.add_trace(go.Scatter(y=df_e, name=log_file_names[df_index][:6], visible=visible), row=1, col=1)
                fig.add_trace(go.Scatter(y=df_c, name=log_file_names[df_index][:6], visible=visible), row=2, col=1)
                fig.add_trace(go.Scatter(y=df_v, name=log_file_names[df_index][:6], visible=visible), row=3, col=1)
                step = dict(args=[{"visible": [False] * 3 * len(list_energy_df)}, ], label=log_file_names[df_index][:6])
                step["args"][0]["visible"][3 * df_index] = True
                step["args"][0]["visible"][3 * df_index + 1] = True
                step["args"][0]["visible"][3 * df_index + 2] = True
                slider_steps.append(step)
            fig.update_layout(sliders=[dict(active=0, pad={"t": 50}, steps=slider_steps, bgcolor="#ffb200",
                                            currentvalue=dict(xanchor="center", font=dict(size=16)))])
            plotly.offline.plot(fig, config={'scrollZoom': True, 'editable': True}, filename=f'{path_output}/DC Voltage.html',
                                auto_open=auto_open_chrome_output)
        else:
            if test_type_dc_voltage_1:
                fig_set_visible_delta = [True] + [False] * 2 * len(alpha_filter)
                fig_set_visible = fig_set_visible_delta * 3
            elif test_type_dc_voltage_2:
                fig_set_visible = [True] * 2 + [False] * fig_set_visible_delta_1 +\
                                  [True] * 2 + [False] * len(voltage_drop_th_steps) +\
                                  [True] * 2 + [False] * fig_set_visible_delta_2
            else:
                fig_set_visible = [True] * 6 + [False] * (fig_set_visible_delta_1 + fig_set_visible_delta_2)
            for index_fig, fig in enumerate(list_of_figs):
                if test_type_dc_voltage_1:
                    slider_steps = []
                    for alpha_filter_index, alpha_filter_trace in enumerate(alpha_filter):
                        step = dict(args=[{"visible": fig_set_visible * plots_per_test}, ],
                                    label=("%.2f" % alpha_filter_trace))
                        for plt_number in range(plots_per_test):
                            index_delta = len(fig_set_visible) * 3 * plt_number
                            step["args"][0]["visible"][index_delta + alpha_filter_index + 1] = True
                            step["args"][0]["visible"][index_delta + len(alpha_filter) + alpha_filter_index + 1] = True
                            step["args"][0]["visible"][index_delta + len(fig_set_visible_delta) + alpha_filter_index + 1] = True
                            step["args"][0]["visible"][index_delta + len(fig_set_visible_delta) + len(alpha_filter) + alpha_filter_index + 1] = True
                            step["args"][0]["visible"][index_delta + 2 * len(fig_set_visible_delta) + alpha_filter_index + 1] = True
                            step["args"][0]["visible"][index_delta + 2 * len(fig_set_visible_delta) + len(alpha_filter) + alpha_filter_index + 1] = True
                        slider_steps.append(step)
                    fig.update_layout(sliders=[dict(active=0, pad={"t": 50}, steps=slider_steps, bgcolor="#ffb200",
                                                    currentvalue=dict(prefix='Alpha filter = ',
                                                                      xanchor="center", font=dict(size=16)))])
                elif test_type_dc_voltage_2:
                    slider_steps = []
                    index_1 = 0
                    index_2 = 0
                    for energy_rise_trace_index, energy_rise_trace in enumerate(energy_rise_th_steps):
                        for current_drop_trace_index, current_drop_trace in enumerate(current_drop_th_steps):
                            for voltage_drop_trace_index, voltage_drop_trace in enumerate(voltage_drop_th_steps):
                                step = dict(args=[{"visible": fig_set_visible * plots_per_test}, ],
                                            label=f'{energy_rise_trace}[dB] @ {current_drop_trace:.2f}[A] @ {voltage_drop_trace:.2f}[V]')
                                for plt_number in range(plots_per_test):
                                    index_delta = len(fig_set_visible) * plt_number
                                    step["args"][0]["visible"][index_delta + 2 + index_1] = True
                                    step["args"][0]["visible"][index_delta + 4 + fig_set_visible_delta_1 + voltage_drop_trace_index] = True
                                    step["args"][0]["visible"][index_delta + 6 + fig_set_visible_delta_1 + len(voltage_drop_th_steps) + index_2] = True
                                index_2 += 1
                                slider_steps.append(step)
                            index_1 += 1
                    fig.update_layout(sliders=[dict(active=0, pad={"t": 50}, steps=slider_steps, bgcolor="#ffb200",
                                                    currentvalue=dict(prefix='Energy Rise TH / Current Drop TH / Voltage Drop TH = ',
                                                                      xanchor="center", font=dict(size=16)))])
                else:
                    slider_steps = []
                    for energy_rise_trace_index, energy_rise_trace in enumerate(energy_rise_th_steps):
                        for current_drop_trace_index, current_drop_trace in enumerate(current_drop_th_steps):
                            step = dict(args=[{"visible": fig_set_visible * plots_per_test}, ],
                                        label=f'{energy_rise_trace}[dB] @ {current_drop_trace:.2f}[A]')
                            for plt_number in range(plots_per_test):
                                index_delta = len(fig_set_visible) * plt_number + 6
                                step["args"][0]["visible"][index_delta + energy_rise_trace_index] = True
                                step["args"][0]["visible"][index_delta + len(energy_rise_th_steps) +
                                                           current_drop_trace_index] = True
                                step["args"][0]["visible"][index_delta + fig_set_visible_delta_1 +
                                                           (len(energy_rise_th_steps) - 1) * energy_rise_trace_index +
                                                           current_drop_trace_index] = True
                            slider_steps.append(step)
                    fig.update_layout(sliders=[dict(active=0, pad={"t": 50}, steps=slider_steps, bgcolor="#ffb200",
                                                    currentvalue=dict(prefix='Energy Rise TH / Current Drop TH = ',
                                                                      xanchor="center", font=dict(size=16)))])
                if plots_per_test == 1:
                    plot_name_updated = log_file_names[index_fig][:-4]
                elif len(list_of_figs) > 1:
                    plot_name_updated = f'{plot_name} {index_fig + 1:03}'
                else:
                    plot_name_updated = plot_name
                fig.update_layout(title=plot_name_updated, title_font_color="#407294", title_font_size=40,
                                  legend_title="Records:", legend_title_font_color="green")
                if add_data_labels:
                    labels = [{"plot": 0, "row": 1, "col": 1, "x": 877}, {"plot": 0, "row": 1, "col": 1, "x": 878},
                              {"plot": 1, "row": 1, "col": 2, "x": 878}, {"plot": 1, "row": 1, "col": 2, "x": 879},
                              {"plot": 2, "row": 1, "col": 3, "x": 877}, {"plot": 2, "row": 1, "col": 3, "x": 878},
                              {"plot": 3, "row": 2, "col": 1, "x": 877}, {"plot": 3, "row": 2, "col": 1, "x": 878},
                              {"plot": 4, "row": 2, "col": 2, "x": 877}, {"plot": 4, "row": 2, "col": 2, "x": 878},
                              {"plot": 5, "row": 2, "col": 3, "x": 877}, {"plot": 5, "row": 2, "col": 3, "x": 878}]
                    for label in labels:
                        x = label["x"]
                        fig.add_annotation(row=label["row"], col=label["col"], x=x, y=plot_list[label["plot"]][x],
                                           text=f'{x}, {round(plot_list[label["plot"]][x], 2)}', showarrow=True, opacity=1)
                if test_type_dc_voltage_2:
                    plotly.offline.plot(fig, config={'scrollZoom': True, 'editable': True},
                                        filename=f'{path_output}/{plot_name_updated} (Vdc {voltage_type_2} of {voltage_type_1}).html',
                                        auto_open=auto_open_chrome_output)
                else:
                    plotly.offline.plot(fig, config={'scrollZoom': True, 'editable': True}, filename=f'{path_output}/{plot_name_updated}.html',
                                        auto_open=auto_open_chrome_output)
    if find_arc_start:
        print('file_name, detection_index')
        for arc_start in list_find_arc_start:
            print('{file_name}, {detection_index}'.format(**arc_start))
    print(f'Python finished... Time: {str(datetime.now())[:-7]}')
    if output_text and inspect.currentframe().f_code.co_name == 'main':
        sys.stdout.close()
        sys.stdout = default_stdout


if __name__ == "__main__":
    main()
