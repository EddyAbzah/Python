import os
import glob
import math
import statistics
from scipy import signal
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from io import StringIO
from my_pyplot import omit_plot as _O, plot as _P, print_chrome as _PC, clear as _PP, print_lines as _PL, send_mail as _SM
import Plot_Graphs_with_Sliders as _G
import my_tools

inverter_type_string = ''
# Use zero based indexing for the columns.
log_with_Vdc = True
log_with_Vdc2 = True
log_with_time_stamp = False
log_delimitation_type = ''
log_arc_detected_string = ''
log_minimum_line_length = 0
log_machine_state_idle = 0
log_machine_state_searching = 0
log_sample_number_column = 0
# log_energy: if the Inverter has a single column → put zero in log_energy_column_number_2:
log_energy_low_column = 0
log_energy_high_column = 0
# log_current: if Inverter is 1ph → put zero in number_2 + number_3:
log_current_ph1_column = 0
log_current_ph2_column = 0
log_current_ph3_column = 0
log_dc_voltage_column = 0
log_power_column = 0
log_state_column_number = 0
# ## print method calls "Getting the ..."
print_method_calls = False


def convert_to_df(log):
    df = pd.DataFrame(data=log)[0]
    return df


def choose_inverter(inverter_string, with_time_stamp):
    """Choose Inverter for lof file parsing."""
    global inverter_type_string
    global log_delimitation_type
    global log_arc_detected_string
    global log_minimum_line_length
    global log_machine_state_idle
    global log_machine_state_searching
    global log_sample_number_column
    global log_with_time_stamp
    global log_energy_low_column
    global log_energy_high_column
    global log_current_ph1_column
    global log_current_ph2_column
    global log_current_ph3_column
    global log_dc_voltage_column
    global log_power_column
    global log_state_column_number
    global log_energy_state
    global log_current_state
    global log_voltage_state
    inverter_type_string = inverter_string.strip()
    if inverter_type_string.lower() == 'jupiterti' or inverter_type_string.lower() == 'jupiter_ti':
        log_delimitation_type = ' '
        log_arc_detected_string = 'ARC_DETECT_DETECTED'
        log_minimum_line_length = 130 + 23 * with_time_stamp
        log_machine_state_idle = 4
        log_machine_state_searching = 0
        log_with_time_stamp = with_time_stamp
        log_sample_number_column = 3 + 2 * with_time_stamp
        log_energy_low_column = 5 + 2 * with_time_stamp
        log_energy_high_column = 4 + 2 * with_time_stamp
        log_current_ph1_column = 8 + 2 * with_time_stamp
        log_current_ph2_column = 10 + 2 * with_time_stamp
        log_current_ph3_column = 12 + 2 * with_time_stamp
        log_dc_voltage_column = 6 + 2 * with_time_stamp
        log_power_column = 7 + 2 * with_time_stamp
        log_state_column_number = 22 + 2 * with_time_stamp
        log_energy_state = None
        log_current_state = None
        log_voltage_state = None
    elif inverter_type_string.lower() == 'jupiter1206':
        log_delimitation_type = ','
        log_arc_detected_string = 'ARC_DETECT_DETECTED'
        log_minimum_line_length = 90 + 23 * with_time_stamp
        log_machine_state_idle = 4
        log_machine_state_searching = 0
        log_with_time_stamp = with_time_stamp
        log_sample_number_column = 1 + 0 * with_time_stamp
        log_energy_low_column = 2 + 0 * with_time_stamp
        log_energy_high_column = 0 + 0 * with_time_stamp
        log_current_ph1_column = 3 + 0 * with_time_stamp
        log_current_ph2_column = 0 + 0 * with_time_stamp
        log_current_ph3_column = 0 + 0 * with_time_stamp
        log_dc_voltage_column = 6 + 0 * with_time_stamp
        log_power_column = 0 + 0 * with_time_stamp
        log_state_column_number = 14 + 0 * with_time_stamp
        log_energy_state = 8 + 0 * with_time_stamp
        log_current_state = 9 + 0 * with_time_stamp
        log_voltage_state = 10 + 0 * with_time_stamp
    elif inverter_type_string.lower() == 'jupiterdsp' or inverter_type_string.lower() == 'jupiter_dsp':
        log_delimitation_type = ','
        log_arc_detected_string = 'ARC_DETECT_DETECTED'
        log_minimum_line_length = 85 + 23 * with_time_stamp - 13 * log_with_Vdc
        log_minimum_line_length = 85
        log_machine_state_idle = 0
        log_machine_state_searching = 1
        log_with_time_stamp = with_time_stamp
        log_sample_number_column = 1 + 0 * with_time_stamp
        log_energy_low_column = 2 + 0 * with_time_stamp
        log_energy_high_column = 0 + 0 * with_time_stamp
        log_current_ph1_column = 3 + 0 * with_time_stamp
        log_current_ph2_column = 0 + 0 * with_time_stamp
        log_current_ph3_column = 0 + 0 * with_time_stamp
        log_dc_voltage_column = 4 + 0 * with_time_stamp
        log_power_column = 5 + 0 * with_time_stamp
        log_state_column_number = 8 + 0 * with_time_stamp
        log_energy_state = 6 + 0 * with_time_stamp
        log_current_state = 7 + 0 * with_time_stamp
        log_voltage_state = 9 + 0 * with_time_stamp
    elif inverter_type_string.lower() == 'venus3dsp' or inverter_type_string.lower() == 'venus3_dsp':
        log_delimitation_type = ','
        log_arc_detected_string = 'ARC_DETECT_DETECTED'
        log_minimum_line_length = 85 + 23 * with_time_stamp
        log_machine_state_idle = 2
        log_machine_state_searching = 0
        log_with_time_stamp = with_time_stamp
        log_sample_number_column = 1 + 0 * with_time_stamp
        log_energy_low_column = 2 + 0 * with_time_stamp
        log_energy_high_column = 0 + 0 * with_time_stamp
        log_current_ph1_column = 3 + 0 * with_time_stamp
        log_current_ph2_column = 0 + 0 * with_time_stamp
        log_current_ph3_column = 0 + 0 * with_time_stamp
        log_dc_voltage_column = 5 + 0 * with_time_stamp
        log_power_column = 6 + 0 * with_time_stamp
        log_state_column_number = 14 + 0 * with_time_stamp
        log_energy_state = None
        log_current_state = None
        log_voltage_state = None
    elif inverter_type_string.lower() == 'venus3' or inverter_type_string.lower() == 'venus3':
        log_delimitation_type = ','
        log_arc_detected_string = 'ARC_DETECT_DETECTED'
        log_minimum_line_length = 85 + 23 * with_time_stamp
        log_machine_state_idle = 0
        log_machine_state_searching = 2
        log_with_time_stamp = with_time_stamp
        log_sample_number_column = 1 + 0 * with_time_stamp
        log_energy_low_column = 2 + 0 * with_time_stamp
        log_energy_high_column = 0 + 0 * with_time_stamp
        log_current_ph1_column = 3 + 0 * with_time_stamp
        log_current_ph2_column = 0 + 0 * with_time_stamp
        log_current_ph3_column = 0 + 0 * with_time_stamp
        log_dc_voltage_column = 5 + 0 * with_time_stamp
        log_power_column = 6 + 0 * with_time_stamp
        log_state_column_number = 14 + 0 * with_time_stamp
        log_energy_state = None
        log_current_state = None
        log_voltage_state = None
    else:
        inverter_type_string = 'Inverter unknown'
        log_delimitation_type = ''
        log_arc_detected_string = ''
        log_minimum_line_length = 0
        log_machine_state_idle = 0
        log_machine_state_searching = 0
        log_sample_number_column = 0
        log_energy_low_column = 0
        log_energy_high_column = 0
        log_current_ph1_column = 0
        log_current_ph2_column = 0
        log_current_ph3_column = 0
        log_dc_voltage_column = 0
        log_power_column = 0
        log_state_column_number = 0
        log_energy_state = 0
        log_current_state = 0
        log_voltage_state = 0
    print('inverter_type_string = ' + inverter_type_string)
    print()
    print('------------------------------------------------------------------------------------------------------')
    print()


def get_files(folder_path, string_filter, skip_if_not_pwr=True):
    """Gets all files with a specific filter: string_filter"""
    log_file_all = []
    log_file_names = []
    log_arc_detections = []
    list_of_files = glob.glob(f'{folder_path}\\*{string_filter}*.log')
    for file_number, file in enumerate(list_of_files):
        if print_method_calls:
            print(f'Getting file number {file_number + 1}')
        file_name = file.split('\\')[-1].lower()
        if skip_if_not_pwr and ('scope' in file_name or 'spi' in file_name or 'mngr' in file_name or 'zes' in file_name):
            continue
        with open(file) as file_before_filter:
            file_after_filter_1 = [line for line in file_before_filter.readlines()
                                   if ('@@' in line and len(line) > log_minimum_line_length)
                                   or log_arc_detected_string in line]
        # if log_with_Vdc:
            # file_after_filter_2 = '\n'.join([line[:line.find(',', 85)] for line in file_after_filter_1 if '@@' in line])
        # else:
        file_after_filter_2 = ''.join([line for line in file_after_filter_1 if '@@' in line])
        log_arc_detections.append([i - 1 for i, s in enumerate(file_after_filter_1) if log_arc_detected_string in s])
        log_file_all.append(pd.read_csv(StringIO(file_after_filter_2), sep=log_delimitation_type, header=None, on_bad_lines='warn', skipinitialspace=True, names=np.arange(30)).dropna(axis=1, how='all'))
        log_file_names.append(os.path.basename(file))
    return log_file_all, log_file_names, log_arc_detections


def get_logs(log_file, test_type, extra_prints):
    """Gets the Energy (in dB), Iac and the Machine State from ALL files in the directory."""
    log_sample_number = log_file[log_sample_number_column].to_numpy().tolist()
    diff = np.diff(log_sample_number)
    if max(diff) > 10:
        bad_samples = np.argwhere(diff > 10).flatten()
        for sample in bad_samples:
            print(f'There is a jump higher than 10 samples from {log_sample_number[sample]} to {log_sample_number[sample + 1]}')
    cut_log_at = 0
    log_energy = []
    log_energy_before_log10 = log_file[log_energy_low_column].to_numpy().T
    if log_energy_high_column != 0:
        for index, value in enumerate(log_file[log_energy_high_column].to_numpy().T):
            log_energy_before_log10[index] = log_energy_before_log10[index] + 2 ** 15 * value
    for index in range(len(log_energy_before_log10)):
        log_energy.append(10 * math.log10(log_energy_before_log10[index]))

    log_current = log_file[log_current_ph1_column].to_numpy().tolist()
    if log_current_ph2_column != 0:
        for index, value in enumerate(log_file[log_current_ph2_column].to_numpy().T):
            log_current[index] = log_current[index] + value
    if log_current_ph3_column != 0:
        for index, value in enumerate(log_file[log_current_ph3_column].to_numpy().T):
            log_current[index] = log_current[index] + value

    log_voltage = log_file[log_dc_voltage_column].to_numpy().tolist()
    log_power = log_file[log_power_column].to_numpy().tolist()
    if not log_with_Vdc:
        log_state = log_file[log_state_column_number].to_numpy().tolist()
    log_energy_before_stage2 = []
    log_current_before_stage2 = []
    log_voltage_before_stage2 = []
    log_power_before_stage2 = []
    if test_type == 1:
        print('Searching the record for an Arc.')
        for index, value in enumerate(log_state):
            if value == log_machine_state_idle or value == log_machine_state_searching:
                log_energy_before_stage2.append(log_energy[index])
                log_current_before_stage2.append(log_current[index])
                log_voltage_before_stage2.append(log_voltage[index])
                log_power_before_stage2.append(log_power[index])
                if index == len(log_state) - 1:
                    cut_log_at = 0
                    print('No Arc is detected in this record; return cut_log_at = 0')
                    break
            else:
                i = 0
                if extra_prints > 0:
                    while i < extra_prints:
                        log_energy_before_stage2.append(log_energy[index + i])
                        log_current_before_stage2.append(log_current[index + i])
                        log_voltage_before_stage2.append(log_voltage[index + i])
                        log_power_before_stage2.append(log_power[index + i])
                        i += 1
                elif extra_prints < 0:
                    del log_energy_before_stage2[extra_prints:]
                    del log_current_before_stage2[extra_prints:]
                    del log_voltage_before_stage2[extra_prints:]
                    del log_power_before_stage2[extra_prints:]
                cut_log_at = index + extra_prints - 15  # removing 15 from "cut_log_at" for better TH calculations
                print('Arc found at index = ' + str(cut_log_at))
                break
    if test_type == 2 or test_type == 3:
        if print_method_calls:
            print(f'Getting the record only for "Search" (Machine State = {log_machine_state_searching}).')
        for index, value in enumerate(log_state):
            if value == log_machine_state_searching:
                log_energy_before_stage2.append(log_energy[index])
                log_current_before_stage2.append(log_current[index])
                log_voltage_before_stage2.append(log_voltage[index])
                log_power_before_stage2.append(log_power[index])
            elif test_type == 3:
                print(f'"omit_arcs_from_record" = True; Arc is at sample number {index}')
                cut_log_at = index
                break
    if test_type == 4 or not log_with_Vdc:
        if print_method_calls:
            print(f'Getting the whole record.')
        log_energy_before_stage2 = log_energy
        log_current_before_stage2 = log_current
        log_voltage_before_stage2 = log_voltage
        log_power_before_stage2 = log_power
        cut_log_at = 0
 #   elif test_type == 5:
    if test_type == 2:
        log_energy_df = pd.DataFrame(data=log_energy_before_stage2)[0]
        log_current_df = pd.DataFrame(data=log_current_before_stage2)[0]
        log_voltage_df = pd.DataFrame(data=log_voltage_before_stage2)[0]
        log_power_df = pd.DataFrame(data=log_power_before_stage2)[0]
    else:
        log_energy_df = pd.DataFrame(data=log_energy)[0]
        log_current_df = pd.DataFrame(data=log_current)[0]
        log_voltage_df = pd.DataFrame(data=log_voltage)[0]
        log_power_df = pd.DataFrame(data=log_power)[0]
    if log_with_Vdc:
        log_state_df = pd.DataFrame(data=[0] * len(log_power))[0]
    else:
        log_state_df = pd.DataFrame(data=log_state)[0]
    log_energy_before_stage2_df = pd.DataFrame(data=log_energy_before_stage2)[0]
    log_current_before_stage2_df = pd.DataFrame(data=log_current_before_stage2)[0]
    log_voltage_before_stage2_df = pd.DataFrame(data=log_voltage_before_stage2)[0]
    log_power_before_stage2_df = pd.DataFrame(data=log_power_before_stage2)[0]
    return (log_energy_df, log_current_df, log_voltage_df, log_power_df, log_state_df, log_energy_before_stage2_df,
            log_current_before_stage2_df, log_voltage_before_stage2_df, log_power_before_stage2_df, cut_log_at)


def get_logs_all(log_file):
    """Gets the Energy (in dB), Iac and the Machine State from ALL files in the directory."""
    all_logs = []
    log_sample_number = log_file[log_sample_number_column].to_numpy().tolist()
    diff = np.diff(log_sample_number)
    if max(diff) > 10:
        bad_samples = np.argwhere(diff > 10).flatten()
        for sample in bad_samples:
            print(f'There is a jump higher than 10 samples from {log_sample_number[sample]} to {log_sample_number[sample + 1]}')
    cut_log_at = 0
    log_energy = []
    log_energy_before_log10 = log_file[log_energy_low_column].to_numpy().T
    if log_energy_high_column != 0:
        for index, value in enumerate(log_file[log_energy_high_column].to_numpy().T):
            log_energy_before_log10[index] = log_energy_before_log10[index] + 2 ** 15 * value
    for index in range(len(log_energy_before_log10)):
        log_energy.append(10 * math.log10(log_energy_before_log10[index]))

    log_current = log_file[log_current_ph1_column].to_numpy().tolist()
    if log_current_ph2_column != 0:
        for index, value in enumerate(log_file[log_current_ph2_column].to_numpy().T):
            log_current[index] = log_current[index] + value
    if log_current_ph3_column != 0:
        for index, value in enumerate(log_file[log_current_ph3_column].to_numpy().T):
            log_current[index] = log_current[index] + value

    all_logs = {'log_energy': pd.DataFrame(data=log_energy)[0]}
    all_logs['log_current'] = pd.DataFrame(data=log_current)[0]
    all_logs['log_voltage'] = pd.DataFrame(data=list(log_file[log_dc_voltage_column]))[0]
    all_logs['log_power'] = log_file[log_power_column].to_numpy().tolist()
    all_logs['log_state'] = log_file[log_state_column_number].to_numpy().tolist()
    all_logs['log_energy_state'] = log_file[log_energy_state].to_numpy().tolist()
    all_logs['log_current_state'] = log_file[log_current_state].to_numpy().tolist()
    all_logs['log_voltage_state'] = log_file[log_voltage_state].to_numpy().tolist()
    if log_with_Vdc2:
        all_logs['log_voltage1'] = pd.DataFrame(data=list(log_file[log_dc_voltage_column - 2]))[0]
        all_logs['log_voltage2'] = pd.DataFrame(data=list(log_file[log_dc_voltage_column - 1]))[0]
    return all_logs


def combine_detections(detection_1, detection_2):
    """for the DC Voltage detection, which has smaller window and filter sizes"""
    if not False in detection_2 or not True in detection_1:
        return detection_1
    diff = np.diff(np.multiply(detection_1, 1))
    detection_indexes_1 = np.argwhere(abs(diff) == 1).flatten()
    if len(detection_indexes_1) % 2 != 0:
        detection_indexes_1 = np.append(detection_indexes_1, len(detection_1))
    diff = np.diff(np.multiply(detection_2, 1))
    detection_indexes_2 = np.argwhere(abs(diff) == 1).flatten()
    if len(detection_indexes_2) % 2 != 0:
        detection_indexes_2 = np.append(detection_indexes_2, len(detection_2))

    new_detection = [False] * len(detection_1)
    for index_1 in range(0, len(detection_indexes_1), 2):
        min_max_time = []
        list_1 = list(range(detection_indexes_1[index_1], detection_indexes_1[index_1 + 1] + 1))
        for index_2 in range(0, len(detection_indexes_2), 2):
            list_2 = list(range(detection_indexes_2[index_2], detection_indexes_2[index_2 + 1] + 1))
            congruence = sorted(list(set(list_1) & set(list_2)))
            if congruence:
                min_max_time.extend([congruence[0], congruence[-1]])
        if min_max_time:
            for detection_index in range(min(min_max_time), max(min_max_time)):
                new_detection[detection_index + 1] = True
    return new_detection


def alpha_beta_filter(df, alpha, stop_if_0=False):
    """Filters a Data Frame with an Alpha / Beta Filter"""
    after_filter = [df[0]]
    for index in range(1, len(df)):
        if stop_if_0 and df[index] == 0:
            break
        after_filter.append(alpha * df[index] + (1 - alpha) * after_filter[index - 1])
    return after_filter


def avg_no_overlap(df, old_sample_rate, new_sample_rate = None):
    """Array to average by sample_ratio or number of samples (example: if old_sample_rate=50Hz & new_sample_rate=50kHz → sample_ratio=1000"""
    after_filter = []
    if new_sample_rate != None:
        samples = int(old_sample_rate / new_sample_rate)    # = sample ratio
    else:
        samples = old_sample_rate
    for i in range(0, len(df), samples):
        sliced = df[i:i + samples]
        after_filter.append(pd.DataFrame.mean(sliced))
    return convert_to_df(after_filter)


def avg_with_overlap(df, jump, overlap_after):
    """Array to average by sample_ratio (example: if old_sample_rate=50Hz & new_sample_rate=50kHz → sample_ratio=1000"""
    after_filter = []
    for i in range(0, len(df) - overlap_after, jump):
        sliced = df[i:i + jump + overlap_after]
        after_filter.append(pd.DataFrame.mean(sliced))
    return convert_to_df(after_filter)


def find_arc_via_dc_voltage(file_name, log_voltage):
    """Find and Arc via the DC Voltage log"""
    diff = np.diff(log_voltage)
    detection_index = np.argwhere(diff == diff.min()).flatten()[0]
    return {"file_name": file_name, "detection_index": detection_index}


def lag_finder(y1, y2, sr):
    n = len(y1)
    corr = signal.correlate(y2, y1, mode='same') / np.sqrt(
        signal.correlate(y1, y1, mode='same')[int(n / 2)] * signal.correlate(y2, y2, mode='same')[int(n / 2)])
    delay_arr = np.linspace(-0.5 * n / sr, 0.5 * n / sr, n)
    delay = delay_arr[np.argmax(corr)]
    print('y2 is ' + str(delay) + ' behind y1')
    plt.figure()
    plt.plot(delay_arr, corr)
    plt.title('Lag: ' + str(np.round(delay, 3)) + ' s')
    plt.xlabel('Lag')
    plt.ylabel('Correlation coeff')
    plt.show()


def get_max_correlation(original, match, start=0, stop=0):
    if stop == 0:
        stop = len(original)
    if start == -1:
        start = len(original) - len(match)
    np_array_original = np.array(original[start:stop])
    np_array_match = np.array(match)
    z = signal.fftconvolve(np_array_original, np_array_match[::-1])
    lags = np.arange(z.size) - (np_array_match.size - 1)
    return start + lags[np.argmax(np.abs(z))]


def remove_adjacent(nums):
    result = []
    for num in nums:
        if len(result) == 0 or num != result[-1]:
            result.append(num)
    return result


def remove_consecutive_duplicates(list, threshold=9223372036854775807):
    result = [list[0]]
    index_threshold = 0
    for index in range(len(list) - 1):
        if list[index + 1] != list[index] or index_threshold > threshold:
            result.append(list[index + 1])
            index_threshold = 0
        else:
            index_threshold += 1
    return pd.Series(result)
