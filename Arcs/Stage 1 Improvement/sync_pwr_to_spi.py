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
import log_spi_or_scope
import log_file
from matplotlib import pyplot
_P = pyplot.plot


# # txt output instead of the console - ATTENTION - if True, there will be no Console output:
output_text = True
path_txt = datetime.now().strftime("%d-%m-%Y %H-%M-%S")
path_txt = f'Terminal Log ({path_txt}).txt'
# # Chrome action for plotly:
plot_offline = True
kill_chrome = False and plot_offline
auto_open_chrome_output = False and plot_offline
# # Inverter stuff:
inverter_type = 'Venus3'
log_with_time_stamp = True
# # Test type:
# # 1 = Regular Arc search (via State Machine).
# # 2 = False Alarms (no Arc search).
# # 3 = False Alarms, BUT, records will be cut manually when the state machine is = 10.
# # 4 = No Filter.
test_type = 4
# # Folders and filters:
path_output = r'M:\Users\HW Infrastructure\PLC team\ARC\Temp-Eddy\F1 Scores\Scores'
path_logs = r'M:\Users\HW Infrastructure\PLC team\ARC\Temp-Eddy\F1 Scores'
path_timing = r'\Scores\Log_PWR_arc_timing.csv'
path_logs_string_filter = 'Rec'
# # this will run the voltage algorithm on SPI records instead of the PWR
take_spi_voltage = True
sample_rate_spi = 50e3
sample_rate_pwr = 35
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
voltage_drop_th_steps = np.arange(0, 0.5, 0.1)
alpha_filter = [a / 100 for a in range(100, 99, -10)]
alpha_filter = [0.0001]
# # set sync_voltage_detection (in "arc_th_calc.py") to 9 if 5;5;3  _  or to 10 if 3;3;2
window_size_2 = 20
filter_size_2 = 15
over_th_limit_2 = 12
sizes_2 = f'W = {window_size_2}; F = {filter_size_2}; TH = {over_th_limit_2}'
sync_pwr_to_spi_bool = False
sync_pwr_to_spi = [165, 220, 225]   # [min, avg, max]
# # set Arc start time:
arc_start_before_time = 10
arc_stop_after_time = 35 * 2
# ## print method calls "detection_on_signals_batch ..."
print_method_calls = True


def main():
    list_of_records = []
    if kill_chrome:
        os.system("taskkill /im chrome.exe /f")
    print(f"Starting Python... Time: {str(datetime.now())[:-7]}")
    arc_timing = pd.read_csv(path_logs + path_timing).sort_values(by='Record')
    arc_timing_dict = arc_timing.set_index('Record').T.to_dict('list')
    log_file.choose_inverter(inverter_string=inverter_type, with_time_stamp=log_with_time_stamp)
    log_files_after_filter, log_file_names, log_arc_detections = list(log_file.get_files(folder_path=path_logs,
                                                                                         string_filter=path_logs_string_filter))
    if take_spi_voltage:
        log_spi_files, log_spi_names, log_scope_files, log_scope_names = list(
            log_spi_or_scope.get_files(folder_path=path_logs, string_filter=path_logs_string_filter))
    print(f'Record,Voltage TH Avg - Energy TH,Voltage TH Min - Energy TH,Voltage TH Avg - Current TH,Voltage TH Min - Current TH')
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
                spi_log_voltage = log_spi_files[index_file]['Vdc']
                spi_down_sampled = log_file.convert_to_df(log_file.alpha_beta_filter(spi_log_voltage, a_filter))
                list_voltage_alpha_filtered.append(log_file.convert_to_df(log_file.avg_no_overlap(
                    spi_down_sampled, sample_rate_spi, sample_rate_pwr)))
            else:
                list_voltage_alpha_filtered.append(log_file.convert_to_df(log_file.alpha_beta_filter(df=log_voltage, alpha=a_filter)))
            voltage_th_list_min.append(arc_th_calc.voltage_algo(list_voltage_alpha_filtered[-1], window_size=window_size_2,
                                                                filter_size=filter_size_2, over_th_limit=over_th_limit_2, true_if_avg=False))
            voltage_th_list_avg.append(arc_th_calc.voltage_algo(list_voltage_alpha_filtered[-1], window_size=window_size_2,
                                                                filter_size=filter_size_2, over_th_limit=over_th_limit_2, true_if_avg=True))
        if not sync_pwr_to_spi_bool:
            skip_samples = 0
        else:
            skip_samples = log_file.get_max_correlation(voltage_th_list_avg[-1], current_th_list, start=-1, stop=-15)
            if skip_samples < sync_pwr_to_spi[0] or skip_samples > sync_pwr_to_spi[2]:
                skip_samples = log_file.get_max_correlation(voltage_th_list_avg[-1], energy_th_list, start=-1, stop=-15)
                if skip_samples < sync_pwr_to_spi[0] or skip_samples > sync_pwr_to_spi[2]:
                    print(f'ERROR!!!! Record = {string_index_file}. skip_samples = {skip_samples}')
                    skip_samples = sync_pwr_to_spi[1]
            if print_method_calls:
                print(f'{string_index_file}: skip_samples = {skip_samples}')
        if plot_offline:
            fig = make_subplots(subplot_titles=['Idc - log', 'Idc - TH', 'Vdc - log', 'Vdc - TH min', 'Vdc - TH avg'],
                                rows=5, cols=1, shared_xaxes=True)
            fig.add_trace(go.Scatter(y=log_current, name='Idc - log'), col=1, row=1)
            fig.add_trace(go.Scatter(y=current_th_list, name='Idc - TH'), col=1, row=2)
            fig.add_trace(go.Scatter(y=list_voltage_alpha_filtered[-1][skip_samples:], name='Vdc - log'), col=1, row=3)
            fig.add_trace(go.Scatter(y=voltage_th_list_min[-1][skip_samples:], name='Vdc - TH min'), col=1, row=4)
            fig.add_trace(go.Scatter(y=voltage_th_list_avg[-1][skip_samples:], name='Vdc - TH avg'), col=1, row=5)
            plotly.offline.plot(fig, config={'scrollZoom': True, 'editable': True}, filename=f'{path_output}/{string_index_file}.html',
                                auto_open=auto_open_chrome_output)


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
    print(f'{energy_rise_th}, {current_drop_th}, {sizes_1}, {f1_score["Voltage Drop"]}, {sizes_2_print}, {avg_or_min}, {sync_print}, {alpha_filter_string}, {f1_score["Precision"]:.2%}, {f1_score["Recall"]:.2%}, {f1_score["F1"]:.2%}')


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
