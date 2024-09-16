import os
import gc
import sys
import inspect
from datetime import datetime
import pandas as pd
import numpy as np
import plotly
from plotly.subplots import make_subplots
import plotly.graph_objects as go
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
# # Folders and filters:
path_output = r'V:\HW_Infrastructure\Analog_Team\ARC_Data\Results\Jupiter\Jupiter+ Improved (7E0872F4-EC)\MANA - Copy'
path_logs = r'V:\HW_Infrastructure\Analog_Team\ARC_Data\Results\Jupiter\Jupiter+ Improved (7E0872F4-EC)\MANA - Copy'
path_timing = r'\Log_PWR_arc_timing.csv'
path_logs_string_filter = 'Rec'
# # this will run the voltage algorithm on SPI records instead of the PWR
take_spi_voltage = True
# ## Venus3 DSP: ## sample_rate_spi = 50e3
sample_rate_spi = 16667
sample_rate_pwr = 28.6
sync_pwr_to_spi_1 = False
sync_via_excel_table = (False, 'SPI delta')
sync_pwr_to_spi_2 = (True, 7500, 0)
sync_pwr_to_spi_3 = [165, 220, 225]   # [min, avg, max]
# # plot name ('.html' ass added later):
plot_name = f'F1 Score test'
# # Energy Rise and Current Drop parameters:
energy_rise_th = 8
current_drop_th = 0.2
window_size_1 = 20
filter_size_1 = 15
over_th_limit_1 = 12
sizes_1 = f'W = {window_size_1}; F = {filter_size_1}; TH = {over_th_limit_1}'
# # Voltage Drop parameters:
voltage_drop_th_steps = np.arange(0, 0.05, 0.0025)
alpha_filter = [0.003, 0.001, 0.0005, 0.0001, 0.00005]
# # set sync_voltage_detection (in "arc_th_calc.py") to 9 if 5;5;3  _  or to 10 if 3;3;2
window_size_2 = 20
filter_size_2 = 15
over_th_limit_2 = 12
sizes_2 = f'W = {window_size_2}; F = {filter_size_2}; TH = {over_th_limit_2}'
# # set Arc start time:
arc_start_before_time = 10
arc_stop_after_time = 35 * 2
# ## print method calls "detection_on_signals_batch ..."
print_method_calls = False


def main():
    if output_text and inspect.currentframe().f_code.co_name == 'main':
        default_stdout = sys.stdout
        if not os.path.exists(path_output):
            os.makedirs(path_output)
        sys.stdout = open(f'{path_output}/{path_txt}', 'w')
    global energy_rise_th
    global current_drop_th
    for energy_rise in [12, 10, 8, 6]:
        for current_drop in [0.2, 0.15, 0.1]:
            energy_rise_th = energy_rise
            current_drop_th = current_drop
            run_main()
            gc.collect()
    if output_text and inspect.currentframe().f_code.co_name == 'main':
        sys.stdout.close()
        sys.stdout = default_stdout


def run_main():
    list_of_records = []
    if kill_chrome:
        os.system("taskkill /im chrome.exe /f")
    if output_text and inspect.currentframe().f_code.co_name == 'main':
        default_stdout = sys.stdout
        if not os.path.exists(path_output):
            os.makedirs(path_output)
        sys.stdout = open(f'{path_output}/{path_txt}', 'w')
    print(f"Starting Python... Time: {str(datetime.now())[:-7]}")
    arc_timing = pd.read_csv(path_logs + path_timing).sort_values(by='Record')
    arc_timing_dict = arc_timing.set_index('Record').T.to_dict('list')
    log_file.choose_inverter(inverter_string=inverter_type, with_time_stamp=log_with_time_stamp)
    log_files_after_filter, log_file_names, log_arc_detections = list(log_file.get_files(folder_path=path_logs,
                                                                                         string_filter=path_logs_string_filter))
    if take_spi_voltage:
        log_spi_files, log_spi_names, log_scope_files, log_scope_names = list(
            log_spi_or_scope.get_files(folder_path=path_logs, string_filter=path_logs_string_filter))
    for index_file, file in enumerate(log_files_after_filter):
        list_voltage_alpha_filtered = []
        voltage_th_list_min = []
        voltage_th_list_avg = []
        string_index_file = f'{log_file_names[index_file][:6]}'
        (log_energy, log_current, log_voltage, log_power, log_state, log_energy_before_stage2, log_current_before_stage2,
         log_voltage_before_stage2, log_power_before_stage2, cut_log_at) = log_file.get_logs(file, test_type=test_type, extra_prints=8)
        energy_th_list = arc_th_calc.plot_all(log_energy_before_stage2, window_size=window_size_1,
                                              filter_size=filter_size_1, over_th_limit=over_th_limit_1)
        current_th_list = arc_th_calc.current_algo_old(log_current_before_stage2, window_size=window_size_1,
                                                       filter_size=filter_size_1, over_th_limit=over_th_limit_1)
        for a_filter in alpha_filter:
            if take_spi_voltage:
                spi_log_voltage = log_spi_files[index_file]['V-DC']
                spi_alpha_filtered = log_file.convert_to_df(log_file.alpha_beta_filter(spi_log_voltage, a_filter))
                if sync_pwr_to_spi_2[0]:
                    spi_down_sampled = log_file.convert_to_df(log_file.avg_no_overlap(spi_alpha_filtered[sync_pwr_to_spi_2[1]:], sample_rate_spi, sample_rate_pwr))
                    if sync_pwr_to_spi_2[2] < 0:
                        list_voltage_alpha_filtered.append(log_file.convert_to_df([0] * abs(sync_pwr_to_spi_2[2]) + list(spi_down_sampled)))
                    else:
                        list_voltage_alpha_filtered.append(spi_down_sampled[sync_pwr_to_spi_2[2]:])
                else:
                    list_voltage_alpha_filtered.append(log_file.convert_to_df(log_file.avg_no_overlap(
                        spi_alpha_filtered, sample_rate_spi, sample_rate_pwr)))
            else:
                list_voltage_alpha_filtered.append(log_file.convert_to_df(log_file.alpha_beta_filter(df=log_voltage, alpha=a_filter)))
            voltage_th_list_min.append(arc_th_calc.voltage_algo(list_voltage_alpha_filtered[-1], window_size=window_size_2,
                                                                filter_size=filter_size_2, over_th_limit=over_th_limit_2, true_if_avg=False))
            voltage_th_list_avg.append(arc_th_calc.voltage_algo(list_voltage_alpha_filtered[-1], window_size=window_size_2,
                                                                filter_size=filter_size_2, over_th_limit=over_th_limit_2, true_if_avg=True))
        if sync_pwr_to_spi_1:
            if sync_via_excel_table[0]:
                skip_samples = arc_timing_dict.get(string_index_file)[2]
            else:
                skip_samples = log_file.get_max_correlation(voltage_th_list_avg[-1], current_th_list, start=-1, stop=-15)
                if skip_samples < sync_pwr_to_spi_3[0] or skip_samples > sync_pwr_to_spi_3[2]:
                    skip_samples = log_file.get_max_correlation(voltage_th_list_avg[-1], energy_th_list, start=-1, stop=-15)
                    if skip_samples < sync_pwr_to_spi_3[0] or skip_samples > sync_pwr_to_spi_3[2]:
                        print(f'ERROR!!!! Record = {string_index_file}. skip_samples = {skip_samples}')
                        skip_samples = sync_pwr_to_spi_3[1]
            if print_method_calls:
                print(f'{string_index_file}: skip_samples = {skip_samples}')
            if skip_samples < 0:
                skip_samples = abs(skip_samples)
                for df_index, df in enumerate(list_voltage_alpha_filtered):
                    list_voltage_alpha_filtered[df_index] = log_file.convert_to_df([0] * skip_samples + list(df))
                for df_index, df in enumerate(voltage_th_list_min):
                    voltage_th_list_min[df_index] = log_file.convert_to_df([0] * skip_samples + list(df))
                for df_index, df in enumerate(voltage_th_list_avg):
                    voltage_th_list_avg[df_index] = log_file.convert_to_df([0] * skip_samples + list(df))
            else:
                for df_index, df in enumerate(list_voltage_alpha_filtered):
                    list_voltage_alpha_filtered[df_index] = df[skip_samples:]
                for df_index, df in enumerate(voltage_th_list_min):
                    voltage_th_list_min[df_index] = df[skip_samples:]
                for df_index, df in enumerate(voltage_th_list_avg):
                    voltage_th_list_avg[df_index] = df[skip_samples:]
        energy_detection_list = energy_th_list >= np.float64(energy_rise_th)
        current_detection_list = current_th_list >= np.float64(current_drop_th)
        list_detection_inverter = log_file.convert_to_df([a and b for a, b in zip(energy_detection_list, current_detection_list)])
        voltage_detection_list = []
        list_detection_inverter_with_voltage_min = []
        list_detection_inverter_with_voltage_avg = []
        for index_a_filter, a_filter in enumerate(alpha_filter):
            for voltage_drop_th in voltage_drop_th_steps:
                if voltage_drop_th == 0:
                    continue
                voltage_detection_list.append(voltage_th_list_min[index_a_filter] >= voltage_drop_th)
                voltage_detection_list.append(voltage_th_list_avg[index_a_filter] >= voltage_drop_th)
                list_detection_inverter_with_voltage_min.append(log_file.convert_to_df(log_file.combine_detections(voltage_detection_list[-2], list_detection_inverter)))
                list_detection_inverter_with_voltage_avg.append(log_file.convert_to_df(log_file.combine_detections(voltage_detection_list[-1], list_detection_inverter)))
        record = {"Record": string_index_file, "Time": arc_timing_dict.get(string_index_file)[1],
                  "Detections - Min": [*list_detection_inverter_with_voltage_min],
                  "Detections - Avg": [*list_detection_inverter_with_voltage_avg],
                  "Detections - Inverter": [energy_detection_list, current_detection_list, list_detection_inverter],
                  "Detections - Voltage": [*voltage_detection_list]}
        list_of_records.append(record)
    print()
    list_f1_scores = []
    if voltage_drop_th_steps[0] == 0:
        list_true_positive = []
        list_false_positive = []
        list_true_negative = []
        list_false_negative = []
        for record in list_of_records:
            true_positives, false_positives, true_negatives, false_negatives = detection_on_signals_batch(
                record["Record"], record["Time"], record["Detections - Inverter"][2])
            list_true_positive.extend(list(true_positives))
            list_false_positive.extend(list(false_positives))
            list_true_negative.extend(list(true_negatives))
            list_false_negative.extend(list(false_negatives))
        f1_score = calculate_f1_score(list_true_positive, list_false_positive, list_true_negative,
                                      list_false_negative)
        f1_score["Voltage Drop"] = 'disabled'
        f1_score["AVG or Min"] = 'disabled'
        f1_score["Alpha Filter"] = 'disabled'
        list_f1_scores.append(f1_score)
        voltage_drop_th_steps_new = voltage_drop_th_steps[1:]
    else:
        voltage_drop_th_steps_new = voltage_drop_th_steps
    for a_filter_index, a_filter in enumerate(alpha_filter):
        for voltage_drop_index, voltage_drop in enumerate(voltage_drop_th_steps_new):
            voltage_drop_string = f'{voltage_drop:.2}'
            avg_or_min_list = ['Detections - Min', 'Detections - Avg']
            for avg_or_min in avg_or_min_list:
                list_true_positive = []
                list_false_positive = []
                list_true_negative = []
                list_false_negative = []
                for record in list_of_records:
                    true_positives, false_positives, true_negatives, false_negatives = detection_on_signals_batch(
                        record["Record"], record["Time"], record[avg_or_min][a_filter_index * len(voltage_drop_th_steps_new) + voltage_drop_index])
                    list_true_positive.extend(list(true_positives))
                    list_false_positive.extend(list(false_positives))
                    list_true_negative.extend(list(true_negatives))
                    list_false_negative.extend(list(false_negatives))
                f1_score = calculate_f1_score(list_true_positive, list_false_positive, list_true_negative, list_false_negative)
                f1_score["Voltage Drop"] = voltage_drop_string
                f1_score["AVG or Min"] = avg_or_min
                f1_score["Alpha Filter"] = f'{a_filter:.02}'
                list_f1_scores.append(f1_score)
    print()
    print()
    print('Old detection algorithm, Detection algorithm with V-DC, Results')
    # print('Energy Rise TH [dB], Current Drop TH [A], Window and Filter, Voltage Drop, Window and Filter, Avg or Min, Detection sync, Alpha Filter, Precision, Recall, F1 Score')
    print('Energy Rise TH [dB], Current Drop TH [A], Window and Filter, Voltage Drop, Window and Filter, Avg or Min, Alpha Filter, Precision, Recall, F1 Score')

    for f1_score in list_f1_scores:
        print_f1_score(f1_score)

    if plot_offline:
        plot_titles = ['Energy Detection', 'Current Detection',
                       *[f'Voltage Detection {vd} {moa}' for vd in voltage_drop_th_steps for moa in ["Min", "Avg"]]]
        specs = np.reshape([[{"secondary_y": False}] for x in range(len(plot_titles))], (len(plot_titles), 1)).tolist()
        for record in list_of_records:
            fig = make_subplots(subplot_titles=plot_titles, rows=len(plot_titles), cols=1, specs=specs, shared_xaxes=True)
            for df_index, df in enumerate(record["Detections"]):
                fig.add_trace(go.Scatter(y=df, name=plot_titles[df_index]), col=1, row=df_index + 1)
            fig.update_layout(title=f'{plot_name} for {energy_rise_th}dB@{current_drop_th}A with VD={[vd for vd in voltage_drop_th_steps]}')
            plotly.offline.plot(fig, config={'scrollZoom': True, 'editable': True}, filename=f'{path_output}/{record["Record"]}.html',
                                auto_open=auto_open_chrome_output)
    print(f'Python finished... Time: {str(datetime.now())[:-7]}')
    if output_text and inspect.currentframe().f_code.co_name == 'main':
        sys.stdout.close()
        sys.stdout = default_stdout


def calculate_f1_score(list_true_positive, list_false_positive, list_true_negative, list_false_negative):
    if len(list_true_positive) == 0:
        precision = 0
        recall = 0
        f1 = 0
    else:
        precision = len(list_true_positive) / (len(list_true_positive) + len(list_false_positive))
        recall = len(list_true_positive) / (len(list_true_positive) + len(list_false_negative))  # a.k.a. sensitivity
        f1 = 2 * ((recall * precision) / (recall + precision))
    f1_scores = {"Precision": precision, "Recall": recall, "F1": f1}
    return f1_scores


def print_f1_score(f1_score):
    if f1_score["Voltage Drop"] == 'disabled':
        avg_or_min = 'disabled'
        alpha_filter_string = 'disabled'
        sizes_2_print = 'disabled'
        sync_print = 'disabled'
    else:
        avg_or_min = f1_score["AVG or Min"][-3:]
        alpha_filter_string = f1_score["Alpha Filter"]
        sizes_2_print = sizes_2
        sync_print = arc_th_calc.sync_voltage_detection
    # print(f'{alpha_filter_string}, {f1_score["Voltage Drop"]}, {avg_or_min}, {f1_score["Precision"]:.2%}, {f1_score["Recall"]:.2%}, {f1_score["F1"]:.2%}')
    # print(f'{energy_rise_th}, {current_drop_th}, {sizes_1}, {f1_score["Voltage Drop"]}, {sizes_2_print}, {avg_or_min}, {sync_print}, {alpha_filter_string}, {f1_score["Precision"]:.2%}, {f1_score["Recall"]:.2%}, {f1_score["F1"]:.2%}')
    print(f'{energy_rise_th}, {current_drop_th}, {sizes_1}, {f1_score["Voltage Drop"]}, {sizes_2_print}, {avg_or_min}, {alpha_filter_string}, {f1_score["Precision"]:.2%}, {f1_score["Recall"]:.2%}, {f1_score["F1"]:.2%}')


def detection_on_signals_batch(record_name, arc_time, detections):
    if print_method_calls:
        print(f'detection_on_signals_batch for record = {record_name}')
    list_true_positives = []
    list_false_positives = []
    list_true_negatives = []
    list_false_negatives = []
    arc_time_start = arc_time - arc_start_before_time
    arc_time_stop = arc_time + arc_stop_after_time

    diff = np.diff(np.multiply(detections, 1))
    detection_indexes = np.argwhere(diff == 1).flatten()
    if arc_time == 0:
        if print_method_calls:
            print('arc_time == 0')
        if len(detection_indexes) == 0:
            list_true_negatives.append(True)
            list_true_positives.append(True)
        else:
            list_false_positives.extend([True] * len(detection_indexes))
    else:
        arc_found = False
        false_alarm = False
        for detection_index in detection_indexes:
            if print_method_calls:
                print(f'detection at = {detection_index}; should be between {arc_time_start} and {arc_time_stop}')
            if detection_index < arc_time_start:
                list_false_positives.append(True)
                false_alarm = True
            elif detection_index <= arc_time_stop:
                list_true_positives.append(True)
                arc_found = True
                break
            else:
                break
        if not arc_found:
            if print_method_calls:
                print(f'Arc not found in this record')
            list_false_negatives.append(True)
        if not false_alarm:
            list_true_negatives.append(True)
        if print_method_calls:
            if arc_found and not false_alarm:
                print(f'record = {record_name} is a perfect record')
            else:
                print(f'record = {record_name} had some issues')
    return list_true_positives, list_false_positives, list_true_negatives, list_false_negatives


if __name__ == "__main__":
    main()
