import os
import statistics
import sys
import glob
from datetime import datetime
from io import StringIO
import pandas as pd
import numpy as np
import plotly
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from matplotlib import pyplot
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
# # Filter PWR records - use the correct case:
inverter_events = {"Sync": 'PLL SYNCED', "Unsync": 'PLL NOT SYNCED', "Stage 1": '<207> ,', "Stage 2": '<15> ,',
                   "Detection": 'ARC_DETECT_DETECTED', "Reset": 'Relay'}
inverter_events = {"Sync": '', "Unsync": 'PLL NOT SYNCED', "Stage 1": '<207>:', "Stage 2": '<15>:',
                   "Detection": 'ARC_DETECT_DETECTED', "Reset": 'Relay'}
inverter_events = {"Sync": 'PLL SYNCED', "Unsync": 'PLL NOT SYNCED', "Stage 1": 'Event [132]', "Stage 2": 'Event [15]',
                   "Detection": 'ARC_DETECT_DETECTED', "Reset": 'Relay'}
string_log_format = ['Power Diff', 'Bitmap', 'Phase Shift', 'Amp Shift', 'Amp Ratio']
string_log_format = ['Power Diff', 'Bitmap', 'Phase Shift', 'Amp Shift', 'Sync flag']
stage2_events_all = ['Time', *string_log_format]
split_record_titles = ['KA1, TH=8', 'KA1, TH=12', 'KA1, TH=16', 'KA2, TH=8', 'KA2, TH=12', 'KA2, TH=16']
record_split_count_1 = 20
record_split_count_2 = 60
# # Folders and filters:
path_output = r'M:\Users\HW Infrastructure\PLC team\ARC\Temp-Eddy'
path_logs = r'V:\HW_Infrastructure\Analog_Team\ARC_Data\Results\Jupiter\Dumbo Jupiter 1206 from Integration (7E1486B4)\FAs Summary 01 to 06 (31-08-2021)\MANA'
path_logs_string_filter = 'm'
log_file_delimiter = ','
# # plot name ('.html' ass added later):
plot_name = f'False Alarms test'
# ## print method calls "string found ..."
print_method_calls = False


def main():
    error_messages = []
    log_file_names = []
    log_events = []
    log_stage2_events = []
    if kill_chrome:
        os.system("taskkill /im chrome.exe /f")
    if output_text:
        default_stdout = sys.stdout
        if not os.path.exists(path_output):
            os.makedirs(path_output)
        sys.stdout = open(f'{path_output}/{path_txt}', 'w')
    print(f"Starting Python... Time: {str(datetime.now())[:-7]}")
    list_of_files = glob.glob(f'{path_logs}\\*{path_logs_string_filter}*.log')

    for file_number, file in enumerate(list_of_files):
        log_file_names.append(os.path.basename(file))
        log_record_events = {key: [] for key in inverter_events.keys()}
        log_record_events["Record"] = file_number + 1
        log_record_stage2_events = {key: [] for key in stage2_events_all}
        log_record_stage2_events["Record"] = file_number + 1
        if print_method_calls:
            print(f'Getting file number {file_number + 1}')
        with open(file) as file_before_filter:
            for line in file_before_filter.readlines():
                for event_type, event_string in inverter_events.items():
                    if event_string in line:
                        log_record_events[event_type].append(line.strip())
                        if event_type == 'Stage 2':
                            date = datetime.strptime(line.strip().split('  <15>')[0], "%d-%m-%Y %H:%M:%S.%f")
                            log_record_stage2_events['Time'].append(date)
                            for index_word, word in enumerate(line.strip().split()[4:]):
                                if index_word == len(string_log_format) - 1:    # because the prints suck
                                    log_record_stage2_events[string_log_format[index_word]].append(float(word[:8]))
                                    break
                                else:
                                    log_record_stage2_events[string_log_format[index_word]].append(float(word))
                        if print_method_calls:
                            print(f'{event_type} event = {line.strip()}')
        log_events.append(log_record_events)
        log_stage2_events.append(log_record_stage2_events)
        # ## check for errors:
        check_1 = len(log_record_events["Sync"]) - len(log_record_events["Stage 1"])
        check_2 = len(log_record_events["Stage 1"]) - len(log_record_events["Stage 2"])
        if check_1 != 0 or check_2 != 0:
            error_messages.append(f'Mismatch between events. Record {file_number + 1}: check_1|2 = {check_1}|{check_2}')
        len_compare = len(log_record_events["Stage 2"])
        if not all(len(ar2) == len_compare for ar2 in [ar1 for ar1 in log_record_stage2_events.values()][:-1]):
            error_messages.append(f'Mismatch in Stage2 events. Record {file_number + 1}: len_compare = {len_compare}')
        if log_record_events["Unsync"]:
            error_messages.append(f'Unsync event. Record {file_number + 1}: count = {len(log_record_events["Unsync"])}')
        if log_record_events["Detection"]:
            error_messages.append(f'Detection event. Record {file_number + 1}: count = {len(log_record_events["Detection"])}')
        if log_record_events["Reset"]:
            error_messages.append(f'Reset event. Record {file_number + 1}: count = {len(log_record_events["Reset"])}')
        if print_method_calls:
            print()
    print('...........................................................................................................')
    print()
    print(f'Record,Sync events,Unsync events,Stage 1 events,Stage 2 events,Arc detections,Resets')
    all_events_1 = {key: 0 for key in inverter_events.keys()}
    for log_number, log_file in enumerate(log_events):
        for key, value in log_file.items():
            if key != 'Record':
                all_events_1[key] += len(value)
        for_print = [log_file[string] for string in inverter_events.keys()]
        print(f'{log_file["Record"]},{",".join([str(len(arr)) for arr in for_print])}')
    print(f'Total,{",".join([str(value) for value in all_events_1.values()])}')
    print()
    print('...........................................................................................................')
    print()
    print(f'Record,Time,Power Diff,Bitmap,Phase Shift,Amplitude Shift,Amplitude Ratio')
    all_records_min_values = {key: 666666666 for key in [*string_log_format]}
    all_records_min_records = {key: '' for key in [*string_log_format]}
    all_records_max_values = {key: -666666666 for key in [*string_log_format]}
    all_records_max_records = {key: '' for key in [*string_log_format]}
    all_records_avg_values_1 = {key: [] for key in [*string_log_format]}
    for log_number, log_file in enumerate(log_stage2_events):
        all_events_2_min = {key: 0 for key in string_log_format}
        all_events_2_max = {key: 0 for key in string_log_format}
        all_events_2_avg = {key: 0 for key in string_log_format}
        # for index_event in range(len(log_file["Time"])):
            # print(f'{log_file["Record"]},{",".join([str(log_file[key][index_event]) for key in stage2_events_all])}')
        for key, value in log_file.items():
            if key != 'Time' and key != 'Record':
                all_events_2_min[key] = min(value)
                if all_events_2_min[key] < all_records_min_values[key]:
                    all_records_min_values[key] = all_events_2_min[key]
                    all_records_min_records[key] = log_file["Record"]
                all_events_2_max[key] = max(value)
                if all_events_2_max[key] > all_records_max_values[key]:
                    all_records_max_values[key] = all_events_2_max[key]
                    all_records_max_records[key] = log_file["Record"]
                all_events_2_avg[key] = statistics.mean(value)
                all_records_avg_values_1[key].extend(value)
        print(f'{log_file["Record"]},Min,{",".join([str(value) for value in all_events_2_min.values()])}')
        print(f'{log_file["Record"]},Max,{",".join([str(value) for value in all_events_2_max.values()])}')
        print(f'{log_file["Record"]},Avg,{",".join([str(value) for value in all_events_2_avg.values()])}')
    print()
    print('...........................................................................................................')
    print()
    all_records_avg_values_2 = {key: statistics.mean(value) for key, value in all_records_avg_values_1.items()}
    print(f',Power Diff,Bitmap,Phase Shift,Amplitude Shift,Amplitude Ratio')
    print(f'Min,{",".join([str(value) for value in all_records_min_values.values()])}')
    print(f'Record,{",".join([str(value) for value in all_records_min_records.values()])}')
    print(f'Max,{",".join([str(value) for value in all_records_max_values.values()])}')
    print(f'Record,{",".join([str(value) for value in all_records_max_records.values()])}')
    print(f'Avg,{",".join([str(value) for value in all_records_avg_values_2.values()])}')
    print()
    print('...........................................................................................................')
    print()
    print('Printing all error messages:')
    for index_message, message in enumerate(error_messages):
        print(f'Error number {index_message + 1}: {message}')
    print('...........................................................................................................')

    if plot_offline:
        plot_columns = 3
        plot_titles = []
        plot_rows = int(len(plot_titles) / plot_columns)
        specs = np.reshape([[{"secondary_y": False}] for x in range(len(plot_titles))], (plot_rows, plot_columns)).tolist()
        fig = make_subplots(subplot_titles=plot_titles, rows=plot_rows, cols=plot_columns, specs=specs, shared_xaxes=True)
        for df_index, df in enumerate(error_messages):
            fig.add_trace(go.Scatter(y=df, name=plot_titles[df_index]), col=1, row=df_index + 1)
        fig.update_layout(title=f'{plot_name}')
        plotly.offline.plot(fig, config={'scrollZoom': True, 'editable': True}, filename=f'{path_output}/mana.html',
                            auto_open=auto_open_chrome_output)
    print(f'Python finished... Time: {str(datetime.now())[:-7]}')
    if output_text:
        sys.stdout.close()
        sys.stdout = default_stdout


if __name__ == "__main__":
    main()
