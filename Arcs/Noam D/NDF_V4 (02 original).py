import os
# import queue
import threading
import time
import asyncio
import pandas as pd
import gc
import numpy as np
import plotly as py
import plotly.graph_objs as go
import math
import statistics
from scipy import signal
from pathlib import Path
from plotly.subplots import make_subplots
EPSILON = 0.000000001
return_min_with_max_avg = False
# que = queue.Queue()

W = '\033[0m'  # white (normal)
R = '\033[31m'  # red
G = '\033[32m'  # green
O = '\033[33m'  # orange
B = '\033[34m'  # blue
P = '\033[35m'  # purpleW = '\033[0m'  # white (normal)
R = '\033[31m'  # red
G = '\033[32m'  # green
O = '\033[33m'  # orange
B = '\033[34m'  # blue
P = '\033[35m'  # purple

import os
import glob
import math
import statistics
from scipy import signal
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from io import StringIO


inverter_type_string = ''
# Use zero based indexing for the columns.
log_with_Vdc = True
log_with_Vdc2 = True
drop_na = False
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


class Sample:
    def __init__(self, x, t):
        self.location = x
        self.time = t

    def __repr__(self):
        return f"Sample({self.location}, {self.time})"


class AlphaBetaFilter:
    def __init__(self, init_sample, alpha=1, beta=0.1, velocity=1):
        self.alpha = alpha
        self.beta = beta
        self.velocity_list = [velocity]
        self.sample_list = [init_sample]
        self.locations = [init_sample.location]
        self.errors = []
        self.predictions = []

    @property
    def last_sample(self):
        return self.sample_list[-1]

    @property
    def last_velocity(self):
        return self.velocity_list[-1]

    def add_sample(self, s: Sample):
        delta_t = s.time - self.last_sample.time
        expected_location = self.predict(delta_t)
        error = s.location - expected_location
        location = expected_location + self.alpha * error
        v = self.last_velocity + (self.beta / delta_t) * error

        # for debugging and results
        self.velocity_list.append(v)
        self.locations.append(location)
        self.sample_list.append(s)
        self.errors.append(error)

    def predict(self, t):
        prediction = self.last_sample.location + (t * self.last_velocity)

        # for debugging and results
        self.predictions.append(prediction)
        return prediction

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
        # log_dc_voltage_column = 5 + 0 * with_time_stamp
        log_dc_voltage_column = 5 + 0 * with_time_stamp
        # log_power_column = 5 + 0 * with_time_stamp
        log_power_column = 6 + 0 * with_time_stamp
        # log_state_column_number = 11 + 0 * with_time_stamp
        log_state_column_number = 11 + 0 * with_time_stamp
        # log_energy_state = 7 + 0 * with_time_stamp
        log_energy_state = 7 + 0 * with_time_stamp
        # log_current_state = 8 + 0 * with_time_stamp
        log_current_state = 8 + 0 * with_time_stamp
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
        if log_with_Vdc:
            cols = [n for n in range(12)]
        else:
            cols = None
        if drop_na:
            log_file_all.append(pd.read_csv(StringIO(file_after_filter_2), sep=log_delimitation_type, header=None, on_bad_lines='warn', skipinitialspace=True, usecols=cols).dropna())
        else:
            log_file_all.append(pd.read_csv(StringIO(file_after_filter_2), sep=log_delimitation_type, header=None, on_bad_lines='warn', skipinitialspace=True, usecols=cols))
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


# <codecell> Import Scope CSV file
def scope_CSV_to_df(path, filename, tag_arr, ch_arr, has_time_col, fs):
    """
        Function import scope data CSV file into Pandas DataFrame

        Inputs:
        `path`              - Data files path (relatively to .py file location); String
        `file_name_arr`     - array of data files you want to analyse; String array
        `tag_arr`           - Array of tags you want to attach to data files; String array
        `ch_arr`            - Array of scope channels you want to analyse; String array

        Returns:
            DF - Pandas Data Frame

        Example of usage :
            df = igf.scope_CSV_to_df(path, file_name_arr, tag_arr, ch_arr)
    """
    tss = time.time()
    # df = pd.read_csv(path + filename + '.csv', header=0)
    df = pd.read_csv(path  , header=0)
    # df = pd.read_csv(path + filename + '.txt', header=0, delimiter='\t')
    #   df=df.add_prefix(tag_arr[0]+'_')
    #   df = df.rename(columns = {df.columns[0]: 'Time'})
    if has_time_col:
        df['Time'] = (df['Time'] - df.loc[0, 'Time'])
    else:
        df['Time'] = df.index * (1 / fs)
    for col in df.columns:
        if 'Unnamed' in col:
            del df[col]

    dt = df.loc[1, 'Time'] - df.loc[0, 'Time']
    Fs = int(1.0 / dt)  # sampling rate
    Ts = 1 / Fs  # sampling interval
    df_len = len(df)
    df_time_len = max(df['Time']) - min(df['Time'])
    tmin = min(df['Time'])
    tmax = max(df['Time'])

    b = 0
    # for filename in file_name_arr[1:] :
    #     df_tmp = pd.read_csv(path+filename+'.csv', header=0)
    #
    #     for col in df_tmp.columns:
    #         if 'Unnamed' in col:
    #             # print(col)
    #             del df_tmp[col]
    #         if 'Time' in col:
    #             # print(col)
    #             del df_tmp[col]
    #     b=b+1
    #
    #     df_tmp=df_tmp.add_prefix(tag_arr[b]+'_')
    #     df[df_tmp.columns] = df_tmp
    #
    # cols=['Time']
    # for s in ch_arr:
    #     c = df.columns[df.columns.str.contains(s)]
    #     cols = cols + c.tolist()
    # df=df[cols]

    temp1 = 'DF Tmin = ' + str(tmin) + '[Sec]; ' + 'DF Tmax = ' + str(tmax) + '[Sec]; \n'
    temp2 = 'DF time length = ' + str(round(df_time_len, 5)) + '[Sec] / ~' + str(
        round(df_time_len / 60, 4)) + '[Min]; \n'
    text = temp1 + temp2 + 'DF length = ' + str(df_len / 1000000) + '[Mega Samples];\n' + 'DF Sampling rate = ' + str(
        round((Fs / 1000), 0)) + '[kSamp/sec]' + '; DF Sampling Interval = ' + str(round((Ts * 1000), 3)) + '[mSec]\n'

    tee = time.time()

    return (df)


# <codecell> Import SPI txt file
def spi_TXT_to_df(path, spi_sample_rate, plot_raw=True):
    """
        (filename, spi_filename,spi_ch_arr, spi_ch_arr, spi_Fs,plot_raw=True)
        Function import SPI data TXT file into Pandas DataFrame adding 'Time' vector and calculates Voltage

        Inputs:
        `path`              - Data files path (relatively to .py file location); String
        `filename`          - file name you want to analyse; String
        `tag_arr`           - Array of tags you want to attach to data files; String array
        `spi_param_arr`     - Array of measured SPI parameters you want to analyse; String array
        `spi_sample_rate`   - SPI sampled data rate [Hz]; integer

        Returns:
            DF - Pandas Data Frame

        Example of usage :
            df_spi = spi_TXT_to_df(path, filename, tag_arr, spi_param_arr, spi_sample_rate)
    """
    tsss = time.time()
    Ts = 1 / spi_sample_rate
    Fs = spi_sample_rate
    df_spi = pd.DataFrame(columns=['Time'])
    # df = pd.read_csv(path + filename + '.txt', header=0)
    df = pd.read_csv(path , header=0)
    i = 0

    for col in df.columns:
        if 'Unnamed' in col:
            del df[col]
        # else:
        #     df = df.rename(columns={col: spi_param_arr[i]})
        #     i = i + 1
    #
    df_spi['Time'] = (df.index) * Ts
    df = df.loc[:, ~df.columns.duplicated()]
    if plot_raw == False:
        for col in df.columns:
            if ('Vin' in col or 'Vout' in col):
                V_quantization = 1 / (2 ** 6)
                df_spi[col] = df[col] * V_quantization
            elif ('Iin' in col or 'IL' in col):
                V_quantization = 1 / (2 ** 9)
                df_spi[col] = df[col] * V_quantization
            else:
                V_quantization = 1 / (2 ** 12)
                V_quantization = 1
                df_spi[col] = df[col] * V_quantization
    else:
        for col in df.columns:
            df_spi[col] = df[col]
    df_len = len(df_spi)
    df_time_len = max(df_spi['Time']) - min(df_spi['Time'])
    tmin = min(df_spi['Time'])
    tmax = max(df_spi['Time'])

    temp1 = 'DF Tmin = ' + str(tmin) + '[Sec]; ' + 'DF Tmax = ' + str(tmax) + '[Sec]; \n'
    temp2 = 'DF time length = ' + str(round(df_time_len, 5)) + '[Sec] / ~' + str(
        round(df_time_len / 60, 4)) + '[Min]; \n'
    text = temp1 + temp2 + 'DF length = ' + str(df_len / 1000000) + '[Mega Samples];\n' + 'DF Sampling rate = ' + str(
        round((Fs / 1000), 0)) + '[kSamp/sec]' + '; DF Sampling Interval = ' + str(round((Ts * 1000), 3)) + '[mSec]'

    teee = time.time()

    return (df_spi)


def kill_chrome(bool):
    if bool:
        os.system("taskkill /im chrome.exe /f")


def df_Chunk(df, t_start, t_end):
    """
    Function cuts data chunk and returns new DF in relevant time frame
    `df`        - Pandas Data Frame
    `t_start`   - Start time
    `t_end`     - End time

    Returns - DataFrame chunk

    Example of usage :
        t_start = 1
        t_end = 5
        df1 = igf.df_Chunk(df, t_start, t_end, dt)spec_res
    """
    start = time.time()
    #df=df.reset_index()
    df = df.reset_index(drop=True)
    dt = df['Time'][0] - df['Time'][1]
    df1 = df[df['Time'] > t_start - dt]
    df1 = df1[df1['Time'] < t_end]
    # df1=df1.astype(float)
    tmin = min(df1['Time'])
    tmax = max(df1['Time'])
    df1_time_len = round(tmax - tmin, 2)
    df1_len = len(df1)

    temp = 'DF "chunk" time length = ' + str(df1_time_len) + '[Sec]/~' + str(round(df1_time_len / 60, 2)) + '[Min];\n'
    text = temp + 'DF "chunk" Start time = ' + str(tmin) + ' [sec]' + '; DF "chunk" End time = ' + str(
        tmax) + ' [sec];\n' + 'DF "chunk" length = ' + str(df1_len / 1000000) + ' [Mega Samples]'
    end = time.time()
    return (df1)


# <codecell> Time Reset
def df_time_reset(dfx, x_col):
    """
        Function resets data chunk time vector

        `dfx`   - Pandas Data Frame
        `x_col` - exact Time col name

        Example of usage :
            df1 = igf.df_time_reset(df1,'Time')
    """
    prop_df = dfx.copy(deep=True)
    prop_df[x_col] = prop_df[x_col] - dfx[x_col].iloc[0]
    return (prop_df)


def df_var_calc(dfi, win_size, x_col):
    """
        Function calculates Rolling Variance of all data frame and return new DF

        `dfi`       - Pandas Data Frame
        `win_size`  - number of sample to calc Variance
        `x_col`     - exact x axis col name

        Example of usage :
            win_size = 10000
            df_var = igf.df_var_calc(df1, win_size, 'Time')
    """
    start = time.time()
    dfi_var = pd.DataFrame()
    dfi_var[x_col] = dfi[x_col]
    col = dfi.columns[1:]
    dfi_var[col] = dfi[col].rolling(window=win_size).var()
    dfi_var = dfi_var.add_suffix('_var')
    dfi_var = dfi_var.rename(columns={dfi_var.columns[0]: x_col})
    # dfi_var[col]=dfi_var.append(dfi_temp)

    end = time.time()

    return (dfi_var)


def df_dB_calc(dfi, x_col):
    """
        Function calculates 20*Log10(dfi) dataframe columns
        `dfi`       - input Pandas Data Frame
        `x_col`     - exact x axis col name

    """
    start = time.time()
    dfi_dB = pd.DataFrame()
    dfi_dB[x_col] = dfi[x_col]
    col = dfi.columns[1:]
    dfi[dfi == 0] = 10 ** -12
    dfi_dB[col] = 20 * np.log10(dfi[col])
    dfi_dB = dfi_dB.add_suffix('_[dB]')
    dfi_dB = dfi_dB.rename(columns={dfi_dB.columns[0]: x_col})
    end = time.time()

    return (dfi_dB)


# <codecell> STFT Transform Calculation function

# <codecell> Plot Data function
def data_plot(dfx, data_name, x_col, max_plot_res, meas_sig, auto_open_plot, file_on, results_path):
    """
        Function genearates Scattergl graph data array and if chose generate html plot
        `dfx`           - input data frame with requested data to plot
        `data name`     - text string to define file and graph name
        `x_col`         - plot X axis column
        `max_plot_res`  - max plot resolution
        `meas_sig`      - requested channel or data column to plot
        `auto_open_plot`- True/False for Auto Open of plots in web browser
        `file_on`       - True/False create html file
        `results_path`  - results path to store html files

        Example of usage :
            (data_out) = igf.data_plot(dfx, data_name, x_col ,max_plot_res, meas_sig, auto_open_plot, file_on, results_path)
    """

    start = time.time()
    plot_res =int(len(dfx) / max_plot_res)
    if plot_res == 0:
        plot_res = 1
    data_out = [x_col]
    # fig_out=[]
    data = []
    for col in dfx:
        if meas_sig in col:
            data.append(
                go.Scattergl(
                    x=dfx[x_col].iloc[::plot_res],
                    y=dfx[col].iloc[::plot_res],
                    name=col + ' - ' + data_name,
                    mode='lines',
                    # mode = 'markers',
                    hoverlabel=dict(namelength=-1)
                )
            )

    fig = go.FigureWidget(data)
    config = {'scrollZoom': True, 'editable': True}
    fig['layout'].update(title=meas_sig + ' ' + data_name)
    fig['layout'].update(xaxis=dict(title=x_col))
    data_out.append(data)
    # fig_out.append(fig)
    txt = results_path + '\\' + meas_sig + ' ' + data_name


    if file_on:
       # fig.write_image(txt + '.jpeg')
        py.offline.plot(fig, auto_open=auto_open_plot, config=config, filename=txt+ '.html')

    end = time.time()

    return (data_out)

def data_plot_streamlit(dfx, added_string, x_col, max_plot_res):
    """
        Function genearates Scattergl graph data array and if chose generate html plot
        `dfx`           - input data frame with requested data to plot
        `data name`     - text string to define file and graph name
        `x_col`         - plot X axis column
        `max_plot_res`  - max plot resolution
        `meas_sig`      - requested channel or data column to plot
        `auto_open_plot`- True/False for Auto Open of plots in web browser
        `file_on`       - True/False create html file
        `results_path`  - results path to store html files

        Example of usage :
            (data_out) = igf.data_plot(dfx, data_name, x_col ,max_plot_res, meas_sig, auto_open_plot, file_on, results_path)
    """

    start = time.time()
    plot_res =int(len(dfx) / max_plot_res)
    if plot_res == 0:
        plot_res = 1
    data_out = [x_col]
    data = []

    for col in dfx:
        if col==x_col:
            continue
        data.append(
            go.Scattergl(
                x=dfx[x_col].iloc[::plot_res],
                y=dfx[col].iloc[::plot_res],
                name=col + ' - ' + added_string,
                mode='lines',
                # mode = 'markers',
                hoverlabel=dict(namelength=-1)
            )
        )

    fig = go.FigureWidget(data)
    config = {'scrollZoom': True, 'editable': True}
    #fig['layout'].update(title= ' ' + added_string)
    #fig['layout'].update(title=' ' + added_string)
    fig['layout'].update(xaxis=dict(title=x_col))
    data_out.append(data)
    # fig_out.append(fig)

    return fig, data_out


# <codecell> Plot N-panes Data function Definition
def data_pane_plot(data_name, data_list, name_list, plot_on, x_sync, tag_arr, channel, results_path):
    """
    Function plot "N" data panes
    N between 1-4
    `data name` - tesxt string to define file and graph name
    `data list` - list of plot data which generated by data_plot()
    `name_list` - list of test strings to define pane names
    `plot_on`   - True/False for Auto Open of plots in web browser
    `x_sync`    - True/False shared_xaxes configuration
    `tag_arr`   - array of filenames
    `channel`    - array of scope channels to be adressed
    `results_path`  - results path to store html files

    Example of usage :
        data_list=[scope_plot_data, var_plot_data, AVG_plot_data]
        name_list=['Scope', 'Var', 'Mean Var']
        igf.data_pane_plot(data_name, data_list, name_list, plot_on, x_sync, tag_arr, channel, results_path)
    """

    start = time.time()
    n_case = len(data_list)

    xaxes_sync = x_sync
    yaxes_sync = False

    spacing = 0.1
    for k in range(1):
        if n_case != len(name_list):
            return ()

        if n_case == 1:
            fig = py.subplots.make_subplots(rows=n_case, cols=1, subplot_titles=(name_list[0]),
                                            specs=[[{"secondary_y": True}]],
                                            shared_xaxes=xaxes_sync, shared_yaxes=yaxes_sync,
                                            vertical_spacing=spacing)
        elif n_case == 2:
            fig = py.subplots.make_subplots(rows=n_case, cols=1, subplot_titles=(name_list[0], name_list[1]),
                                            specs=[[{"secondary_y": True}], [{"secondary_y": True}]],
                                            shared_xaxes=xaxes_sync, shared_yaxes=yaxes_sync,
                                            vertical_spacing=spacing)
        elif n_case == 3:
            fig = py.subplots.make_subplots(rows=n_case, cols=1,
                                            subplot_titles=(name_list[0], name_list[1], name_list[2]),
                                            specs=[[{"secondary_y": True}], [{"secondary_y": True}],
                                                   [{"secondary_y": True}]],
                                            shared_xaxes=xaxes_sync, shared_yaxes=yaxes_sync,
                                            vertical_spacing=spacing)
        elif n_case == 4:
            fig = py.subplots.make_subplots(rows=n_case, cols=1,
                                            subplot_titles=(name_list[0], name_list[1], name_list[2], name_list[3]),
                                            specs=[[{"secondary_y": True}], [{"secondary_y": True}],
                                                   [{"secondary_y": True}], [{"secondary_y": True}]],
                                            shared_xaxes=xaxes_sync, shared_yaxes=yaxes_sync,
                                            vertical_spacing=spacing)
        else:
            return ()

        n = 1
        for trace in data_list:
            loop_count = len(trace[k + 1])
            for i in range(loop_count):
                fig.add_trace(trace[k + 1][i], n, 1, secondary_y=False)
                fig.update_xaxes(title_text=trace[0], row=n, col=1)
            n = n + 1

        config = {'scrollZoom': True, 'editable': True}
        fig['layout'].update(title=data_name + ': ' + results_path.split('\\')[-1])
        import os
        this_dir = os.path.dirname(os.path.abspath(__file__))
        os.chdir(Path(this_dir))
        txt = results_path + '\\' + data_name
        py.offline.plot(fig, auto_open=plot_on, config=config, filename=txt + '.html')

    end = time.time()




def goertzel_calc(df, Freqs_for_gorezel_arr, ch_arr, Fixed_point,After_ds_Fs,N,alpha):
    """
        Function calculates spectromgram with SciPy "stft" function
        Inputs:
            `Z_arr`             - Tuplet of RAW magnitudes maps in dBm
            `Z_name_arr`        - Array of Z names in relevant order
            `t`                 - time vector
            `f`                 - frequency vector
            `zero_span_arr`     - list of frequencoes to perform Zero Span Calc
            `ch_arr`            - list of mesured scope channels
        Outputs:
            `df_fft`            - Zero Span results Pandas Data Frame
        Example of usage :
            (df_fft, df_MH, df_AVG, t, f, Z_arr, Name_arr) = igf.df_spectrogram_calc(dfx, fft_win, fft_win_overlap, zero_span_arr, ch_arr)
    """
    config = {'scrollZoom': True}
    fig=go.FigureWidget()
    for col in ch_arr:
        for detect_freq in Freqs_for_gorezel_arr:
            Time_array = df['Time'].to_numpy()
            RX_array = df[col].to_numpy()
            if Fixed_point:

                samp = GoertzelSampleBySampleFixedpoint(detect_freq, After_ds_Fs, N, alpha)
            else:
                RX_array = RX_array / (2 ** 12)
                samp = GoertzelSampleBySample(detect_freq, After_ds_Fs, N, alpha=alpha)

            dft_out_goertzel_1_TONE = []
            Time_axis_1_TONE = []
            for idx, sample in enumerate(RX_array):
                temp = samp.process_sample(sample)
                if temp is not None:
                    dft_out_goertzel_1_TONE.append(samp.ErgFiltered)
                    Time_axis_1_TONE.append(Time_array[idx])
                    samp.reset()
            str(int(detect_freq / 1000)) + '[kHZ]'
            dft_out_goertzel_with_iir_db = 10 * np.log10((np.array(dft_out_goertzel_1_TONE)))
            fig.add_trace(go.Scattergl(x=Time_axis_1_TONE[:],
                                       y=dft_out_goertzel_with_iir_db[:],
                                       name='Gorezel ' + str(col) + ' FS=' +str(int(After_ds_Fs / 1000))+ '[kHZ]'  + " " + str(
                                           int(detect_freq / 1000)) + '[kHZ]', mode="lines",
                                       visible=True,
                                       showlegend=True))
    return fig




def df_mean_calc(dfi, win_size, x_col):
    """
        Function calculates rolling average of all data frame and return new DF

        `dfi`       - Pandas Data Frame
        `win_size`  - number of sample to rolling average
        `x_col`     - exact x axis col name

        Example of usage :
            df_mean = igf.df_mean_calc(df_fft, 100, 't')
    """
    start = time.time()
    dfi_mean = pd.DataFrame()
    dfi_mean[x_col] = dfi[x_col]
    col = dfi.columns[1:]
    dfi_mean[col] = dfi[col].rolling(window=win_size).mean()
    dfi_mean = dfi_mean.add_suffix('_mean')
    dfi_mean = dfi_mean.rename(columns={dfi_mean.columns[0]: x_col})
    # dfi_var[col]=dfi_var.append(dfi_temp)

    end = time.time()

    return (dfi_mean)


# <codecell> STFT Transform Calculation function
def  df_stft_RAW_calc(dfx, Fs, fft_win, fft_win_overlap, ch_arr):
    """
        Function calculates spectromgram with SciPy "stft" function
        Inputs:
            `dfx`               - Pandas Data Frame
            `Fs`                - Data sampling frequency
            `fft_win'           - fft window size
            `fft_win_overlap`   - exact x axis col name
            `ch_arr`            - list of mesured scope channels
        Outputs:
            `t`                 - time vector
            `f`                 - frequency vector
            `Zdata`             - Tuplet of RAW magnitudes maps in dBm
            `Name_arr`          - Array of Z names in relevant order
        Example of usage :
             f, t, Zxx = signal.stft(sig, Fs, window=w, nperseg=N, noverlap=fft_win_overlap)
    """
    start = time.time()
    Z_arr = []
    Name_arr = []
    print_flag = 0
    for meas_sig in ch_arr:
        N = fft_win  # Number of point in the fft
        w = signal.hamming(N)  # FFT window
        sig = dfx[meas_sig]

        f, t, Zxx = signal.stft(sig, Fs, window=w, nperseg=N, noverlap=fft_win_overlap)
        Z_arr.append(Zxx)
        Name_arr.append(meas_sig)
        del Zxx
        # w = signal.blackman(N) #FFT window
        # for col_fft in dfx.columns[:]:
        #     if meas_sig in col_fft:
        #         sig = dfx[col_fft]
        #
        #         f, t, Zxx = signal.stft(sig, Fs, window=w, nperseg=N, noverlap=fft_win_overlap)
        #         Z_arr.append(Zxx)
        #         Name_arr.append(col_fft)
        #         del Zxx
    gc.collect()
    end = time.time()

    return (t, f, Z_arr, Name_arr)


# <codecell> Z Magnitude Calculation function
def Z_mag_calc(Zraw_arr):
    """
        Function calculates magnitudes of STFT output
        Inputs:
            `Zraw_arr`          - Tuplet of comples STFT results matrices
        Outputs:
            `ZdBm_arr`          - Tuplet of STFT magnitudes in dBm
        Example of usage :
            ZdBm_arr = igf.Z_mag_calc(Zraw_arr)
    """
    start = time.time()

    ZdBm_arr = []
    for i in range(len(Zraw_arr)):
        Zxx = Zraw_arr[i]
        Zxx = 2 * np.abs(Zxx)
        Zxx[Zxx == 0] = 10 ** -12
        Zxx = (20 * np.log10(Zxx / 10)) - 30
        ZdBm_arr.append(Zxx)
        del Zxx
        gc.collect()
    end = time.time()

    return (ZdBm_arr)


# <codecell> Z Phase Calculation function
def Z_phase_calc(Zraw_arr, phase_unwrap):
    """
        Function calculates spectromgram with SciPy "stft" function
        Inputs:
            `Zraw_arr`          - Tuplet of comples STFT results matrices
        Outputs:
            `Zphase_arr`          - Tuplet of STFT magnitudes in dBm
        Example of usage :
            Zphase_arr = igf.Z_phase_calc(Zraw_arr)
    """
    start = time.time()
    Zphase_arr = []
    for i in range(len(Zraw_arr)):
        Zxx = Zraw_arr[i]
        Zxx = np.angle(Zxx, deg=True)
        # Zphase_arr.append(Zxx)
        if phase_unwrap:
            Zxx = np.unwrap(Zxx)
        Zphase_arr.append(Zxx)
        del Zxx
        gc.collect()
    end = time.time()

    return (Zphase_arr)


def scpectrogram_plot(Z, t, f, max_plot_res, fmax, fmin, t_start, t_stop, plot_on, results_path, meas_sig):
    """
        Function generates spectrogram plot

        `Z`             - Results matrix
        `t`             - time vector      (series)
        `f`             - frequency vector (series)
        `max_plot_res`  - max plot resolution (float)
        `fmax`          - max freq (float)
        `fmin`          - min freq (float)
        `t_start`       - chunk start time (float)
        `plot_on`       - True/False for Auto Open of plots in web browser (boolean)
        `results_path`  - Test resutls path (string)
        `meas_sig`      - presented signal name (string)
    """
    start = time.time()
    f_min_ind = int(fmin / (f[1] - f[0]))
    f_max_ind = int(fmax / (f[1] - f[0])) + 1
    spec_res = int(len(t) / max_plot_res)

    if spec_res == 0:
        spec_res = 1
    trace = [go.Heatmap(x=t[::spec_res], y=f[f_min_ind:f_max_ind],
                        z=Z[f_min_ind:f_max_ind, ::spec_res], colorscale='Jet')]

    layout = go.Layout(title='Spectrogram [dBm]: ' + results_path.split('\\')[-1] if meas_sig == '' else meas_sig,
                       yaxis=dict(title='Frequency [Hz]'),  # x-axis label
                       xaxis=dict(title='Time [sec]'),  # y-axis label
                       )

    fig = go.Figure(data=trace, layout=layout)
    config = {'scrollZoom': True, 'editable': True}
    if meas_sig == '':
        meas_sig = '01 Spectrogram'
    txt = results_path + '\\' + meas_sig +' Spectrogram ' + '.html'
    #txt = results_path + meas_sig +' Spectrogram ' + '.html'
    # fig.write_image(txt + '.jpeg')
    py.offline.plot(fig, auto_open=plot_on, config=config, filename=txt)
    end = time.time()
    print(f'finished Spectrogram+ ; length = {end - start}')


def spectrogram_plot_for_streamlit(Z, t, f, max_plot_res, fmax, fmin, meas_sig):
    """
        Function generates spectrogram plot

        `Z`             - Results matrix
        `t`             - time vector      (series)
        `f`             - frequency vector (series)
        `max_plot_res`  - max plot resolution (float)
        `fmax`          - max freq (float)
        `fmin`          - min freq (float)
        `t_start`       - chunk start time (float)
        `plot_on`       - True/False for Auto Open of plots in web browser (boolean)
        `results_path`  - Test resutls path (string)
        `meas_sig`      - presented signal name (string)
    """
    start = time.time()
    f_min_ind = int(fmin / (f[1] - f[0]))
    f_max_ind = int(fmax / (f[1] - f[0])) + 1
    print('------------------')
    print('for meassig = ', meas_sig)
    print(F"len of t={len(t)}")
    print(F"max plot res is ={max_plot_res}")
    spec_res = int(len(t) / max_plot_res)
    print(F"Spacing is: spec_res={spec_res})")
    print('------------------')



    if spec_res == 0:
        print(F"Spacing is 0 ploting full res")
        spec_res = 1

    trace = [go.Heatmap(x=t[::spec_res], y=f[f_min_ind:f_max_ind],
                        z=Z[f_min_ind:f_max_ind,::spec_res], colorscale='Jet')]

    layout = go.Layout(title='Spectrogram [dBm]: ' + meas_sig,
                       yaxis=dict(title='Frequency [Hz]'),  # x-axis label
                       xaxis=dict(title='Time [sec]'),  # y-axis label
                       )

    fig = go.Figure(data=trace, layout=layout)
    config = {'scrollZoom': True, 'editable': True}
    return fig,trace
    # if meas_sig == '':
    #     meas_sig = '01 Spectrogram'
    #txt = results_path + '\\' + meas_sig +' Spectrogram ' + '.html'
    #txt = results_path + meas_sig +' Spectrogram ' + '.html'
    # fig.write_image(txt + '.jpeg')
    # py.offline.plot(fig, auto_open=plot_on, config=config, filename=txt)
    # end = time.time()
   # print(f'finished Spectrogram+ ; length = {end - start}')



def sort_by_rec(file):
    return file[file.find('Rec'):file.find('Rec') + 6]

def Sort_folder(lab_folder):
    Spi_files=[]
    Scope_files=[]
    for dirPath,foldersInDir,fileName in os.walk(lab_folder):
        if fileName is not []:
            for file in fileName:
                file=file.lower()
                if file.endswith('.txt'):
                    if 'spi' in file:
                        loc = os.sep.join([dirPath,file])
                        Spi_files.append(loc)
                if file.endswith('.csv'):
                    if 'scope' in file :
                        loc = os.sep.join([dirPath,file])
                        Scope_files.append(loc)
    return sorted(Spi_files, key=sort_by_rec),sorted(Scope_files, key=sort_by_rec)


def Avg_no_overlap(arr, N):
    # ARR TO AVERGAE BY N SAMPLES
    # ids = np.arange(len(arr)) // N
    # out = np.bincount(ids, arr) / np.bincount(ids)
    # return(out)
    out = []
    N = int(N)
    for i in range(0, len(arr), N):
        sliced = arr[i:N + i]
        out.append(statistics.mean(sliced))

        if (len(sliced)) < N:
            break
    return np.array(out)


# <codecell> Spliding Spectrum Calculation Function
def sliding_spectrum(Z, t, f, win_size, win_overlap, meas_sig):
    """
    Function performing sliding MaxHold and AVG functios on Spectrogram and returns data frame of spectrum frames

    `Z`             - Results matrix
    `t`             - time vector      (series)
    `f`             - frequency vector (series)
    `win_size`      - number of samples to calculate Max hold and AVG (integer)
    `win_overlap`   - number of samples to overlap windows (integer)
    `meas_sig`      - presented signal name (string)

    Example of usage:
        (df_MH, df_AVG, t_list)=igf.sliding_spectrum(Z, t, f, win_size, win_overlap, meas_sig)
    """

    start = time.time()
    if win_size > len(t) - 1:
        print("win size is to long! Please reduce the number of samples to be below " + str(len(t)) + " value")
        return ()

    if win_size < win_overlap:
        print(
            "win overlap size is to long! Please reduce the number of samples to be below " + str(win_size) + " value")
        return ()

    df_MH = pd.DataFrame()
    df_AVG = pd.DataFrame()
    df_MIN = pd.DataFrame()

    # df_MH = pd.DataFrame(columns=['f'])
    # df_AVG = pd.DataFrame(columns=['f'])
    # df_MH['f']=f
    # df_AVG['f']=f

    win_size = int(win_size)
    win_overlap = int(win_overlap)

    # tstep=t[1]-t[0]
    i1 = 0
    i2 = win_size

    di = win_size - win_overlap
    if di == 0:
        di = 1;
    t_list = t[i2::di]
    while i2 < len(t):
        Zt = Z[:, i1:i2].copy()
        Zt_AVG = Zt.mean(axis=1)
        Zt_MH = Zt.max(axis=1)
        if return_min_with_max_avg:
            Zt_MIN = Zt.min(axis=1)
        # Zt_MH=Zt.std(axis=1)

        t1 = t[i1]
        t2 = t[i2]
        col_name = meas_sig + ' @ t = ' + str(round(t1, 3)) + '-' + str(round(t2, 3)) + ' [S]'


        #working methos raises error of PerformanceWarning 'de-fragmented frame'
        Zt_MH_copy=Zt_MH.copy()
        Zt_AVG_copy=Zt_AVG.copy()
        df_MH[col_name] = Zt_MH_copy
        df_AVG[col_name] = Zt_AVG_copy
        del Zt_MH_copy
        del Zt_AVG_copy


        # Should work without error but slow as fuckkkk

        # df_temp_MH = pd.DataFrame({col_name: Zt_MH})
        # df_temp_AVG = pd.DataFrame({col_name: Zt_AVG})
        #
        # df_MH = df_MH.assign(y=df_temp_MH)
        # df_MH.rename({'y': col_name}, axis=1, inplace=True)
        # df_AVG = df_MH.assign(y=df_temp_AVG)
        # df_AVG.rename({'y': col_name}, axis=1, inplace=True)
        #
        if return_min_with_max_avg:
            df_MIN[col_name] = Zt_MIN

        i1 = i1 + di
        i2 = i2 + di

    end = time.time()
    if return_min_with_max_avg:
        return df_MH, df_AVG, t_list, df_MIN
    else:
        return df_MH, df_AVG, t_list, 0


# <codecell> Calculates Sliding DWT Coeeficients from data Frame col
def Sliding_WDT(dfx, x_col, data_col, win_size, overlap, waveletname, decomp_level):
    """
        Function calculates WDT coeeficients and exports it as DataFrame

    """
    start = time.time()
    if win_size < overlap: raise ValueError('Window size must be larger than overlap')

    overlap_ratio = overlap / win_size

    N = NextPowerOfTwo(win_size)
    win_size = int(2 ** N)
    overlap = int(overlap_ratio * win_size)
    txt = '\nNormalization to next power of 2 \n ==> new win_size = %i,  new overlap = %i\n' % (win_size, overlap)

    ds = win_size - overlap
    idx_start = 0
    idx_end = int(win_size - 1)
    Fs = round(1 / (dfx[x_col].iloc[1] - dfx[x_col].iloc[0]))
    t_array = dfx[x_col].iloc[idx_end::ds]

    df_wavelet = pd.DataFrame(columns=[x_col, data_col])
    df_wavelet[x_col] = t_array
    l = len(t_array)
    avg_data = np.zeros(l)
    avg_coeffs = np.zeros((decomp_level, l))

    df_col_name = []

    data = dfx[data_col].to_numpy()
    k = 0
    while idx_end < len(dfx):
        data_segment = data[idx_start:idx_end]
        idx_start = idx_start + ds
        idx_end = idx_start + win_size - 1

        seg = np.abs(data_segment)
        seg = np.power(seg, 2)
        avg_data[k] = np.mean(seg)

        coeffs = pywt.wavedec(data_segment, waveletname, mode='zero', level=decomp_level)

        i = decomp_level
        j = 1

        while i > 0:
            di = coeffs[i]
            di = np.abs(di)
            di = np.power(di, 2)
            avg_coeffs[j - 1, k] = np.mean(di)
            if k == 0:
                fi_h_kHz = round((Fs / (2 ** j)) / 1000, 1)
                fi_l_kHz = round((Fs / (2 ** (j + 1))) / 1000, 1)
                if j <= decomp_level:
                    d_col = 'wdt_d' + str(j) + ' @ ' + str(((fi_l_kHz))) + '[kHz]-' + str((fi_h_kHz)) + '[kHz]'
                else:
                    d_col = 'wdt_a' + str(j - 1) + ' @ ' + str(0) + '[kHz] - ' + str(fi_l_kHz) + '[kHz]'
                df_col_name.append(d_col)
            i = i - 1
            j = j + 1

        k = k + 1
    df_wavelet[data_col] = avg_data.tolist()
    for i in range(0, decomp_level):
        df_wavelet[df_col_name[i]] = avg_coeffs[i].tolist()

    end = time.time()
    # df_wavelet=df_wavelet.reset_index()
    return (df_wavelet)


# <codecell> Goertzel (ZeroSpan DFT) function



def find_nearest(arr, target):
    # Find the index of the nearest value in the array to the target value
    idx = (np.abs(arr - target)).argmin()
    return idx

def ZeroSpan_calc(Z_arr, Z_name_arr, t, f, zero_span_arr, ch_arr):
    """
        Function calculates spectromgram with SciPy "stft" function
        Inputs:
            `Z_arr`             - Tuplet of RAW magnitudes maps in dBm
            `Z_name_arr`        - Array of Z names in relevant order
            `t`                 - time vector
            `f`                 - frequency vector
            `zero_span_arr`     - list of frequencoes to perform Zero Span Calc
            `ch_arr`            - list of mesured scope channels
        Outputs:
            `df_fft`            - Zero Span results Pandas Data Frame
        Example of usage :
            (df_fft, df_MH, df_AVG, t, f, Z_arr, Name_arr) = igf.df_spectrogram_calc(dfx, fft_win, fft_win_overlap, zero_span_arr, ch_arr)
    """
    start = time.time()
    df_fft = pd.DataFrame(columns=['t'])
    for meas_sig in ch_arr:
        flag = 0
        f_arr = zero_span_arr
        z_ind = 0
        for col_fft in Z_name_arr:
            Z_dBm = Z_arr[z_ind]
            z_ind = z_ind + 1
            if meas_sig in col_fft:
                for current_freq in f_arr:
                    ind=np.where(f==current_freq)
                    if ind is []:
                        ind=ind[0][0]
                    else:
                        ind=find_nearest(f,current_freq)

                   # ind = int(ff / (f[1] - f[0]))
                    z_dBm = (Z_dBm[ind, :])
                    df_fft[col_fft + ' @ ' + str(round(float(f[ind] / 1000), 1)) + ' [kHz]'] = z_dBm
                    # print(str(ind))

                if flag == 0:
                    flag = 1
                    df_fft['t'] = t

    end = time.time()

    return (df_fft)

def goertzel(samples, sample_rate, *freqs):
    """
    Implementation of the Goertzel algorithm, useful for calculating individual
    terms of a discrete Fourier transform.

    `samples` is a windowed one-dimensional signal originally sampled at `sample_rate`.

    The function returns 2 arrays, one containing the actual frequencies calculated,
    the second the coefficients `(real part, imag part, power)` for each of those frequencies.
    For simple spectral analysis, the power is usually enough.

    Example of usage :

        # calculating frequencies in ranges [400, 500] and [1000, 1100]
        # of a windowed signal sampled at 44100 Hz

        freqs, results = goertzel(some_samples, 44100, (400, 500), (1000, 1100))
    """
    window_size = len(samples)
    f_step = sample_rate / float(window_size)
    f_step_normalized = 1.0 / window_size

    # Calculate all the DFT bins we have to compute to include frequencies
    # in `freqs`.
    bins = set()
    for f_range in freqs:
        f_start, f_end = f_range
        k_start = int(math.floor(f_start / f_step))
        k_end = int(math.ceil(f_end / f_step))

        if k_end > window_size - 1: raise ValueError('frequency out of range %s' % k_end)
        bins = bins.union(range(k_start, k_end))

    # For all the bins, calculate the DFT term
    n_range = range(0, window_size)
    freqs = []
    results = []
    for k in bins:

        # Bin frequency and coefficients for the computation
        f = k * f_step_normalized
        w_real = 2.0 * math.cos(2.0 * math.pi * f)
        w_imag = math.sin(2.0 * math.pi * f)

        # Doing the calculation on the whole sample
        d1, d2 = 0.0, 0.0
        for n in n_range:
            y = samples[n] + w_real * d1 - d2
            d2, d1 = d1, y

        # Storing results `(real part, imag part, power)`
        results.append((
            0.5 * w_real * d1 - d2, w_imag * d1,
            d2 ** 2 + d1 ** 2 - w_real * d1 * d2)
        )
        freqs.append(f * sample_rate)
    return freqs, results


# <codecell> Sliding Goertzel (ZS DFT) calculation
def sliding_goertzel(dfx, t_col_name, sig_col_name, FFT_freq, win_size, win_overlap):
    """
    Function performing sliding Goertzel functios on RAW data frame and returns Goertzel for specific frequency

    `dfx`           - input Data Frame --> Column from DF
    `t_col_name`    - name of time vector in data frame  (str)
    `sig_col_name`  - presented signal name/scope channel (string)
    `FFT_freq'      - value of required FFT freq.
    `win_size`      - number of samples to calculate FFT window (integer)
    `win_overlap`   - number of samples to overlap windows (integer)

    (results_df) = igf.sliding_goertzel(dfx, t_col_name, sig_col_name, FFT_freq, win_size, win_overlap)
    """

    start = time.time()

    # get signal vector and sample rate
    sx = dfx[sig_col_name].to_numpy()

    # get time vector and sample rate
    t = dfx[t_col_name].to_numpy()
    sample_rate = 1 / (t[2] - t[1])

    if win_size > len(t) - 1:
        print("win size is to long! Please reduce the number of samples to be below " + str(len(t)) + " value")
        return ()

    if win_size < win_overlap:
        print(
            "win overlap size is to long! Please reduce the number of samples to be below " + str(win_size) + " value")
        return ()

    # calculate frequency resolution
    f_resolution = sample_rate / win_size
    f1 = FFT_freq + (f_resolution / 2)
    f2 = FFT_freq + (f_resolution)
    results_df = pd.DataFrame(columns=['t', 'data'])

    win_size = int(win_size)
    win_overlap = int(win_overlap)

    i1 = 0
    i2 = win_size
    di = win_size - win_overlap
    t_list = t[i2::di]
    results_df['t'] = t_list
    df_ind = 0
    while i2 < len(t):
        sxt = sx[i1:i2]
        freqs, results = goertzel(sxt, sample_rate, (f1, f2))
        res_P = results[0][2]
        results_df.loc[df_ind, 'data'] = res_P
        df_ind = df_ind + 1
        i1 = i1 + di
        i2 = i2 + di

    col_name = sig_col_name + ' @ f = ' + str(round((freqs[0] / 1000), 2)) + ' [kHz]'
    results_df = results_df.rename(columns={'t': 'Time', 'data': col_name})

    end = time.time()

    return (results_df)


# <codecell> Multiple Pairs ZeroSpan Covariance and Correlation function
def ZeroSpan_Correlator(dfx, t_col_name, win_size, freq_pairs):
    """
        Function calculates covariance and correlation on 2 ZeroSpan results
        Inputs:
            `dfx`               - Input data frame
            `t_col_name`    - name of time vector in data frame  (str)
            `win_size`          - number of samples of covariance/correlation calcs
            `freq_pairs`        - a tuplet of source vector pairs [[30.0,60.0],[70.0, 30.0],[16.6, 50.0],...], each pair is a str series with partial vector names
        Outputs:
            `df_pairs`          - dataframe with time and analysed signals pairs
            `df_corr`           - dataframe with time and corralation results
            `df_cov`            - dataframe with time and covariance results
    """
    df_pairs = pd.DataFrame(columns=[t_col_name])
    df_corr = pd.DataFrame(columns=[t_col_name])
    df_cov = pd.DataFrame(columns=[t_col_name])
    df_pairs[t_col_name] = dfx[t_col_name]
    df_corr[t_col_name] = dfx[t_col_name]
    df_cov[t_col_name] = dfx[t_col_name]

    for fp in freq_pairs:
        for col in dfx.columns:
            if fp[0] in col and 'mean' not in col:
                df_pairs[col] = dfx[col]
                a = col
            if fp[1] in col and 'mean' not in col:
                df_pairs[col] = dfx[col]
                b = col

        c = a + ' vs ' + fp[1] + ' [kHz]'
        df_corr[c] = df_pairs.rolling(window=win_size).corr().unstack()[a][b]
        df_cov[c] = df_pairs.rolling(window=win_size).cov().unstack()[a][b]

    return (df_pairs, df_corr, df_cov)


# <codecell> AFCI_Stage1_Functions
def AFCI_Stage_1(dfx, t_col_name, WindowSize, noise_method, FilterSize, Threshold, K):
    """
        Function calculates covariance and correlation on 2 ZeroSpan results
        Inputs:
            `dfx`           - Input data frame
            `t_col_name`    - name of time vector in data frame  (str)
            `noise_win`     - number of samples of noise filter
            `noise_method`  - string, avg/max/min
            `signal_win`    - number of samples of noise filter

        Outputs:
            `df_SNR`        - SNR dataframe with time and analysed signals

    """
    df_STG1 = pd.DataFrame(columns=[t_col_name])
    df_STG1[t_col_name] = dfx[t_col_name]

    for col in dfx.columns[1:]:
        if 'mean' not in col and t_col_name not in col and 'Sample' not in col:
            SampleIndex = WindowSize + FilterSize + 20  # Skipping the first 20 samples
            signal = dfx[col].to_numpy()
            detection = np.zeros(len(signal))
            while (SampleIndex < len(signal)):
                i = SampleIndex - WindowSize
                if noise_method == 'max':
                    Noise = max(signal[(SampleIndex - WindowSize - FilterSize):i])
                if noise_method == 'min':
                    Noise = min(signal[(SampleIndex - WindowSize - FilterSize):i])
                if noise_method == 'avg':
                    Noise = np.average(signal[(SampleIndex - WindowSize - FilterSize):i])

                OverThresholdCounter = 0
                while (i < SampleIndex):
                    sample = signal[i]
                    if (abs(sample - Noise) > Threshold):
                        OverThresholdCounter += 1
                    if (OverThresholdCounter >= K):
                        detection[i] = 1
                        break
                    i += 1
                SampleIndex += 1

            df_STG1[col] = detection.tolist()
    return (df_STG1)


# <codecell> Find next power of 2
def NextPowerOfTwo(number):
    """
    # Returns next power of two following 'number'
    """
    return np.ceil(np.log2(number))


# <codecell> Extend numpy array to next power of 2 samples
def Extend_to_NextPowerOfTwo(arr, pad_mode='constant'):
    """
    # Returns next power of two following 'number'

    """
    nextPower = NextPowerOfTwo(len(arr))
    deficit = int(math.pow(2, nextPower) - len(arr))
    extended_arr = np.pad(arr, (0, deficit), mode=pad_mode)
    return extended_arr


#######from ndf old ######


def make_fig(row,col,subplot_titles,shared_xaxes):
    all_specs = np.array([[{"secondary_y": True}] for x in range(row * col)])

    all_specs_reshaped = (np.reshape(all_specs, (col, row)).T).tolist()

    fig = make_subplots(rows=row, cols=col,
                        specs=all_specs_reshaped, shared_xaxes=True, subplot_titles=subplot_titles)
    return fig

def plot_with_slider(df,lab_folder,config,auto_open,kill_chrome_bool,channels,row, col,plots_per_pane,filename_out,max_res):
    fig = make_fig(row=1, col=1, subplot_titles=channels, shared_xaxes=True)
    Rec_arr = []
    plot_spaces = int(len(df) / max_res)
    if plot_spaces == 0:
        plot_spaces = 1
    for col in df.columns:
        if col in channels:
            fig.add_trace(go.Scattergl(x=df[col][::plot_spaces],
                                       y=df[col][::plot_spaces],
                                       name=col + " " ,
                                       visible=False,
                                       showlegend=True), row=1, col=1)

    resutls_path = lab_folder + '\\' + 'kaka' + '\\'
    if not os.path.exists(resutls_path):
        os.makedirs(resutls_path)
    fig.update_layout(
        title=os.path.basename(lab_folder),
        font_family="Courier New",
        font_color="blue",
        font_size=20,
        title_font_family="Times New Roman",
        title_font_color="red",
        legend_title_font_color="green"
    )
    for i in range(plots_per_pane):
        fig.data[i].visible = True
    steps = []
    for i in range(0, int(len(fig.data) / plots_per_pane)):
        Temp = channels[i]

        step = dict(
            method="update",
            args=[{"visible": [False] * len(fig.data)},
                  {"title": "Slider  switched to step "}],
            label=str(Temp)  # layout attribute
        )
        j = i * plots_per_pane
        for k in range(plots_per_pane):
            step["args"][0]["visible"][j + k] = True
        steps.append(step)
    sliders = [dict(
        active=10,
        currentvalue={"prefix": "REC: "},
        pad={"t": 50},
        steps=steps
    )]
    fig.update_layout(
        sliders=sliders)

    # py.offline.plot(fig, config=config, auto_open=auto_open,
    #                 filename=resutls_path + os.path.basename(lab_folder) + filename_out + '.html')
    # kill_chrome(kill_chrome_bool)
    return fig


def ndf_data_plot(dfx, data_name, x_col, max_plot_res, auto_open_plot, file_on, results_path):
    """
        Function genearates Scattergl graph data array and if chose generate html plot
        `dfx`           - input data frame with requested data to plot
        `data name`     - text string to define file and graph name
        `x_col`         - plot X axis column
        `max_plot_res`  - max plot resolution
        `meas_sig`      - requested channel or data column to plot
        `auto_open_plot`- True/False for Auto Open of plots in web browser
        `file_on`       - True/False create html file
        `results_path`  - results path to store html files

        Example of usage :
            (data_out) = igf.data_plot(dfx, data_name, x_col ,max_plot_res, meas_sig, auto_open_plot, file_on, results_path)
    """

    start = time.time()
    plot_res = int(len(dfx) / max_plot_res)
    if plot_res == 0:
        plot_res = 1
    data_out = [x_col]
    # fig_out=[]

    data = []
    for col in dfx:
        data.append(
            go.Scattergl(
                x=dfx[x_col].iloc[::plot_res],
                y=dfx[col].iloc[::plot_res],
                name=col + ' - ' + data_name,
                mode='lines',
                # mode = 'markers',
                hoverlabel=dict(namelength=-1)
            )
        )

    fig = go.FigureWidget(data)
    config = {'scrollZoom': True, 'editable': True}

    fig['layout'].update(xaxis=dict(title=x_col))
    data_out.append(data)
    # fig_out.append(fig)
    txt = results_path + '/' + ' ' + data_name
    if file_on:
        py.offline.plot(fig, auto_open=auto_open_plot, config=config, filename=txt + '.html')

    end = time.time()

    return (data_out)


def npz_to_array(path):
    data = np.load(path + '/mat.npz')

    return data.f.t, data.f.f, data.f.Zraw


def Save_Zraw(results_path, t, f, Zraw):
    np.savez(results_path + '\mat.npz', t=t, f=f, Zraw=Zraw)
    retur


def Save_df_fft_mag(results_path, df_fft, test_name):
    df_fft.to_csv(results_path + '/' + test_name + ' df_fft_mag.csv', index=False, header=True)


def Save_df_fft_phase(results_path, df_fft, test_name):
    df_fft.to_csv(results_path + '/' + test_name + ' df_fft_phase.csv', index=False, header=True)


def df_dBTOv_calc_all(dfi):
    """
        Function calculates 20*Log10(dfi) dataframe columns
        `dfi`       - input Pandas Data Frame
        `x_col`     - exact x axis col name

    """
    start = time.time()

    dfi_V = pd.DataFrame()
    col = dfi.columns[1:]
    # dfi[dfi==240]=10**-12
    dfi_V[col] = np.exp(dfi[col] / 20)
    # dfi_V=dfi_V.rename(columns={dfi_V.columns[0]: x_col})
    end = time.time()

    return (dfi_V)


def df_dBTOv_calc(dfi, x_col):
    """
        Function calculates 20*Log10(dfi) dataframe columns
        `dfi`       - input Pandas Data Frame
        `x_col`     - exact x axis col name

    """
    start = time.time()

    dfi_V = pd.DataFrame()
    dfi_V[x_col] = dfi[x_col]
    col = dfi.columns[1:]
    # dfi[dfi==240]=10**-12
    dfi_V[col] = np.exp(dfi[col] / 20)
    dfi_V = dfi_V.rename(columns={dfi_V.columns[0]: x_col})
    end = time.time()

    return (dfi_V)


def Read_df_fft(path):
    df_fft = pd.read_csv(path, header=0)  # Reading the df_fft from the  LV450 Basic AFD Evaluation V4.0
    df_fft = df_fft[df_fft['t'] > 0.9]  # removing junk
    return df_fft


def SNR(df_fft, zero_span_arr, results_path, plot, factor, string):
    df_fft = df_fft[df_fft['t'] > 0.9]
    t = df_fft[['t']]
    win_size = 10
    data = []
    SNR = pd.DataFrame()
    for col in df_fft.columns[1:]:
        # plt.plot(t,df_fft[col])
        # plt.show()
        Signal = df_dBTOv_calc(df_fft, col)  # in VOLTS

        Noise = Signal[col].rolling(window=win_size).std().median() / np.sqrt(win_size)
        # print(Noise)
        AVG = Signal[col].rolling(window=win_size).mean().median()
        # print(AVG)
        TEMP = (abs(Signal[col] - AVG)) / Noise
        TEMP[TEMP == 0] = 10 ** -12
        SNR[col] = 20 * np.log10(TEMP)

        SNR[col] = SNR[col].rolling(window=100).mean()
        sp = np.fft.fft(np.sin(t))
        freq = np.fft.fftfreq(t.shape[-1])

        data.append(
            go.Scattergl(
                x=t['t'],
                y=SNR[col],
                mode='lines',
                name=col,
                # mode = 'markers',
                hoverlabel=dict(namelength=-1)
            )
        )
        Signal = []
        Noise = 0
        AVG = 0

    fig = go.FigureWidget(data)
    config = {'scrollZoom': True, 'editable': True}
    fig['layout'].update(title=string + 'SNR')
    fig['layout'].update(xaxis=dict(title='time'))
    fig['layout'].update(yaxis=dict(title='SNR[dB]'))
    # data_out.append(data)
    results_path = results_path + '/' + str(factor)
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    py.offline.plot(fig, auto_open=plot, config=config,
                    filename=results_path + '/' + string + ' SNR ds by ' + str(factor) + ', the df_fft is ' + str(
                        round(zero_span_arr[0] / 1000, 2)) + '[Khz].html')
    return SNR, t['t']


def MH_plot_for_gui(res_name_arr, ZdBm, t, f, MH_time, Overlap_time, name, Factor, ch_arr, plot, results_path):
    t_resolution = 0.001
    win_size = int(MH_time / t_resolution)
    win_overlap = int(Overlap_time / t_resolution)
    max_plot_res = 100000

    indices = [i for i, elem in enumerate(res_name_arr)]

    df_MH_1 = pd.DataFrame(columns=['f'])
    df_AVG_1 = pd.DataFrame(columns=['f'])

    df_MH_1['f'] = f
    df_AVG_1['f'] = f
    for i in indices:
        meas_sig = res_name_arr[i]
        (df_MH_temp, df_AVG_temp, t_list) = sliding_spectrum(ZdBm[i], t, f, win_size, win_overlap, meas_sig)

        df_MH_1[df_MH_temp.columns] = df_MH_temp
        df_AVG_1[df_AVG_temp.columns] = df_AVG_temp

        del df_MH_temp
        del df_AVG_temp
    file_on = False
    plot_on = False
    MH1_plot_data = data_plot(df_MH_1, 'Sliding Spectrum MH', 'f', max_plot_res, ch_arr[0], plot_on, file_on,
                              results_path)

    plot_on = True
    data_list = [MH1_plot_data]

    name_list = ['Sliding MH Spectrum dBm']
    x_sync = True
    data_name = str(Factor) + name + ' Sliding FFT MH'
    data_pane_plot(data_name, data_list, name_list, plot_on, x_sync, ch_arr, ch_arr[0], results_path)


def MH_plot(res_name_arr, ZdBm, t, f, MH_time, Overlap_time, name, Factor, ch_arr, plot, results_path):
    t_resolution = 0.001
    win_size = int(MH_time / t_resolution)
    win_overlap = int(Overlap_time / t_resolution)
    max_plot_res = 100000

    indices = [i for i, elem in enumerate(res_name_arr)]

    df_MH_1 = pd.DataFrame(columns=['f'])
    df_AVG_1 = pd.DataFrame(columns=['f'])

    df_MH_1['f'] = f
    df_AVG_1['f'] = f
    for i in indices:
        meas_sig = res_name_arr[i]
        (df_MH_temp, df_AVG_temp, t_list) = sliding_spectrum(ZdBm[i], t, f, win_size, win_overlap, meas_sig)

        df_MH_1[df_MH_temp.columns] = df_MH_temp
        df_AVG_1[df_AVG_temp.columns] = df_AVG_temp

        del df_MH_temp
        del df_AVG_temp
    file_on = False
    plot_on = False
    MH1_plot_data = data_plot(df_MH_1, 'Sliding Spectrum MH', 'f', max_plot_res, ch_arr[0], plot_on, file_on,
                              results_path)

    tag_arr = ['VRX']
    plot_on = True
    data_list = [MH1_plot_data]

    name_list = ['Sliding MH Spectrum dBm']
    x_sync = True
    data_name = str(Factor) + ' ' + name + ' Sliding FFT MH and AVG Spectrum'
    data_pane_plot(data_name, data_list, name_list, plot_on, x_sync, tag_arr, ch_arr[0], results_path)


def SNR_plus(df_fft, zero_span_arr, results_path, plot, factor, string):
    df_fft = df_fft[df_fft['t'] > 0.9]
    t = df_fft[['t']]
    win_size = 10
    data = []
    data_out = [t]
    SNR = pd.DataFrame()
    Signal_in_time = pd.DataFrame()
    Signal = df_dBTOv_calc_all(df_fft)
    for col in df_fft.columns[1:]:
        for one_freq in zero_span_arr:
            if str(one_freq) in col:
                # plt.plot(t,df_fft[col])
                # plt.show()
                # Signal=df_dBTOv_calc(df_fft,col)# in VOLTS
                window = signal.gaussian(10, 5)
                plt.plot(Signal[col])
                plt.show()
                Signal_in_time[col] = np.convolve(Signal[col], window)
                # Signal=df_dBTOv_calc(df_fft,col)# in VOLTS
                plt.plot(Signal_in_time)
                plt.show()

                Noise = Signal_in_time[col].rolling(window=win_size).std().median() / np.sqrt(win_size)
                AVG = Signal_in_time[col].rolling(window=win_size).mean().median()
                TEMP = ((abs(Signal_in_time[col] - AVG)) / Noise)
                TEMP[TEMP == 0] = 10 ** -12
                SNR[col] = 20 * np.log10(TEMP)

                data.append(
                    go.Scattergl(
                        x=t['t'],
                        y=SNR[col],
                        mode='lines',
                        name=col,
                        # mode = 'markers',
                        hoverlabel=dict(namelength=-1)
                    )
                )
                Signal_in_time = []
                Noise = 0
                AVG = 0

    fig = go.FigureWidget(data)
    config = {'scrollZoom': True, 'editable': True}
    # fig['layout'].update(title=name)
    # fig['layout'].update(xaxis=dict(title = t))
    data_out.append(data)

    py.offline.plot(fig, auto_open=True, config=config, filename=results_path + '/' + 'df_fft.html')
    return data_out


def SNR_Matced(df_fft, zero_span_arr, results_path, plot, factor, string):
    df_fft = df_fft[df_fft['t'] > 0.9]
    t = df_fft[['t']]
    win_size = 10
    data = []
    data_out = [t]
    Signal_in_time = pd.DataFrame()
    SNR = pd.DataFrame()
    for col in df_fft.columns[1:]:
        # plt.plot(t,df_fft[col])
        # plt.show()
        Signal = df_dBTOv_calc(df_fft, col)  # in VOLTS
        window = signal.gaussian(10, 0.05)
        # plt.plot(Signal[col])
        # plt.show()
        Signal_in_time[col] = np.convolve(Signal[col], window)
        Noise = Signal[col].rolling(window=win_size).std().median() / np.sqrt(win_size)
        AVG = Signal[col].rolling(window=win_size).mean().median()
        TEMP = (abs(Signal_in_time[col] - AVG)) / Noise
        TEMP[TEMP == 0] = 10 ** -12
        SNR[col] = 20 * np.log10(TEMP)
        SNR[col]

        sp = np.fft.fft(np.sin(t))
        freq = np.fft.fftfreq(t.shape[-1])

        data.append(
            go.Scattergl(
                x=t['t'],
                y=SNR[col],
                mode='lines',
                name=col,
                # mode = 'markers',
                hoverlabel=dict(namelength=-1)
            )
        )
        Signal = []
        Noise = 0
        AVG = 0

    fig = go.FigureWidget(data)
    config = {'scrollZoom': True, 'editable': True}
    fig['layout'].update(title=string + 'SNR')
    fig['layout'].update(xaxis=dict(title='time'))
    fig['layout'].update(yaxis=dict(title='SNR[dB]'))
    # data_out.append(data)
    results_path = results_path + '/' + str(factor)
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    py.offline.plot(fig, auto_open=plot, config=config,
                    filename=results_path + '/' + string + ' SNR ds by ' + str(factor) + ', the df_fft is ' + str(
                        round(zero_span_arr[0] / 1000, 2)) + '[Khz].html')
    return SNR, t['t']


def Had_telem(df_fft, zero_span_arr, results_path, win_size):
    df_fft = df_fft[df_fft['t'] > 1]
    t = df_fft[['t']]
    df_xcor = pd.DataFrame(columns=['t'])
    df_xcor['t'] = df_fft['t']
    win_size = 10
    data_snr = []
    data_out_snr = [t]
    data = []
    data_out = [t]
    SNR = pd.DataFrame()
    DIFF = pd.DataFrame()
    for col in df_fft.columns[1:]:
        for one_freq in zero_span_arr:
            if str(one_freq) in col:
                # plt.plot(t,df_fft[col])
                # plt.show()
                Signal = df_dBTOv_calc(df_fft, col)  # in VOLTS
                Signal.to_csv(results_path + '/' + ' signal.csv', index=False, header=True)
                Noise = Signal[col].rolling(window=win_size).std().median() / np.sqrt(win_size)
                AVG = Signal[col].rolling(window=win_size).mean().median()
                TEMP = (abs(Signal[col] - AVG)) / Noise
                TEMP.to_csv(results_path + '/' + ' temp.csv', index=False, header=True)
                TEMP[TEMP == 0] = 10 ** -12
                SNR[col] = 20 * np.log10(TEMP)
                SNR.to_csv(results_path + '/' + ' snr.csv')
                AVG_SNR = SNR[col].rolling(window=win_size).mean().median()
                TOP_SNR = AVG_SNR * 2
                DIFF = SNR[col]

                len_of_sig = (len(DIFF))
                i = 0
                while i < len_of_sig:  # using the snr defien where is the telem!
                    temp = DIFF[i:i + 5].values.tolist()
                    res = all(i < j for i, j in zip(temp, temp[1:]))
                    if res:
                        sample = DIFF.iloc[i]
                        if sample > TOP_SNR:
                            # print (i)
                            is_telem = statistics.mean(DIFF[i:i + 200].values.tolist())
                            if is_telem > 35:  # print('this is telem')
                                i = i + 300

                    i += 1
                # Signal=[]
                Noise = 0
                AVG = 0

    # df = pd.DataFrame(Signal)
    SNRK = SNR.rolling(window=500).mean()
    # COV=Signal.rolling(window=win_size).cov().unstack()#.to_csv(results_path+'/' +' COV1.csv')
    COV = Signal.rolling(window=win_size).cov().unstack()  # .to_csv(results_path+'/' +' COV1.csv')
    # NUMPT_COV=COV.to_numpy()
    # for col in COV:
    #     tempt=COV[col]

    for col in COV.columns:
        data.append(
            go.Scattergl(
                x=t['t'],
                y=COV[col],
                mode='lines',
                name=col[0] + col[1],
                # mode = 'markers',
                hoverlabel=dict(namelength=-1)
            )
        )

    fig = go.FigureWidget(data)
    config = {'scrollZoom': True, 'editable': True}
    # fig['layout'].update(title=name)
    # fig['layout'].update(xaxis=dict(title = t))
    data_out.append(data)
    py.offline.plot(fig, auto_open=True, config=config, filename=results_path + '/' + 'COV_fft.html')

    # py.offline.plot(fig,auto_open = False, config=config, filename=results_path+'/'+'df_fft.html')
    return data_out, COV


def stage_1_energy_raise(EnergyDB, WindowSize, FilterSize, OverThresholdLimit):
    EnergyThresholdList = []
    EnergyThresholdList.append(0)
    SampleIndex = WindowSize + FilterSize + 20  # Skipping the first 20 samples because of Inverter noises.
    while (SampleIndex < len(EnergyDB)):
        EnergyThreshold = 4;
        MinFilterWindow = min(EnergyDB[(SampleIndex - WindowSize - FilterSize):SampleIndex - WindowSize])
        while EnergyThreshold < 50:
            OverThresholdCounter = 0
            i = SampleIndex - WindowSize
            while (i < SampleIndex):

                if EnergyDB.iloc[i] > (MinFilterWindow + EnergyThreshold + 1):
                    # if MinFilterWindow-EnergyDB.iloc[i]>(EnergyThreshold+1):
                    OverThresholdCounter += 1
                if (OverThresholdCounter >= OverThresholdLimit):
                    break
                i += 1
            if (OverThresholdCounter < OverThresholdLimit):
                break
            EnergyThreshold += 0.5
        if EnergyThreshold == 4:
            EnergyThreshold = 0
        else:
            EnergyThresholdList.append(EnergyThreshold)
        SampleIndex += 1
    return max(EnergyThresholdList)


def stage_1_Iac_raise(Iac_arr, WindowSize, FilterSize, OverThresholdLimit):
    Iac_ThresholdList = []
    Iac_ThresholdList.append(0)
    SampleIndex = WindowSize + FilterSize + 20  # Skipping the first 20 samples because of Inverter noises.
    while (SampleIndex < len(Iac_arr)):
        MaxCurrentDrop = 0.1;
        MinFilterWindow = (Iac_arr[(SampleIndex - WindowSize - FilterSize):SampleIndex - WindowSize]).mean().iloc[0]
        while MaxCurrentDrop < 1:
            OverThresholdCounter = 0
            i = SampleIndex - WindowSize
            while (i < SampleIndex):

                if MinFilterWindow - Iac_arr.iloc[i].values[0] > (+MaxCurrentDrop):
                    # if MinFilterWindow-Iac_arr.iloc[i]>(EnergyThreshold+1):
                    OverThresholdCounter += 1
                if (OverThresholdCounter >= OverThresholdLimit):
                    break
                i += 1
            if (OverThresholdCounter < OverThresholdLimit):
                break
            MaxCurrentDrop += 0.05
        if MaxCurrentDrop == 0.1:
            MaxCurrentDrop = 0
        else:
            Iac_ThresholdList.append(MaxCurrentDrop)
        SampleIndex += 1
    return max(Iac_ThresholdList)


def cheak_harmonics(f_resampling, fpeak, f_fft, k, good_k):
    for i in range(1, k + 1):
        if (abs(f_fft + 1000) < abs(f_resampling - i * fpeak) or abs(f_resampling - i * fpeak) < abs(f_fft - 1000)):
            good_k += 1
    return good_k


def Get_downsampled_signal(x, fs, target_fs, order, Lpf_type):
    decimation_ratio = np.round(fs / target_fs)
    if fs < target_fs:
        raise ValueError("Get_downsampled_signal")
    else:
        try:
            if Lpf_type == 'None':
                y0 = x[::int(decimation_ratio)]
            else:
                y0 = signal.decimate(x, int(decimation_ratio), order, zero_phase=True, ftype=Lpf_type)
            # y1 = signal.decimate(y0,2, 2,zero_phase=True,ftype='iir')
            # f_poly = signal.resample_poly(y, 100, 20)
        except:
            print('error in ds func!')
        actual_fs = fs / decimation_ratio
    return y0, actual_fs


def Get_downsampled_signal_NO_FILTER(x, fs, target_fs):
    decimation_ratio = np.round(fs / target_fs)
    if fs < target_fs:
        raise ValueError("Get_downsampled_signal")
    else:
        try:
            y0 = x[::int(decimation_ratio)]
            # y1 = signal.decimate(y0,2, 2,zero_phase=True,ftype='iir')
            # f_poly = signal.resample_poly(y, 100, 20)
        except:
            y0 = x[::int(decimation_ratio)]
        actual_fs = fs / decimation_ratio
    return y0, actual_fs


def poly_resample(psg, new_sample_rate, old_sample_rate):
    return signal.resample_poly(psg, new_sample_rate, old_sample_rate, axis=0)

def alpha_beta_filter(df, alpha):
    """Filters a Data Frame with an Alpha / Beta Filter"""
    after_filter = [df[0]]
    for index in range(1, len(df)):
        after_filter.append(alpha * df[index] + (1 - alpha) * after_filter[index - 1])
    return after_filter

def spi_TXT_to_df_for_gui(inputFile, tag_arr, spi_sample_rate):
    """
        Function import SPI data TXT file into Pandas DataFrame adding 'Time' vector and calculates Voltage

        Inputs:
        `path`              - Data files path (relatively to .py file location); String
        `filename`          - file name you want to analyse; String
        `tag_arr`           - Array of tags you want to attach to data files; String array
        `spi_param_arr`     - Array of measured SPI parameters you want to analyse; String array
        `spi_sample_rate`   - SPI sampled data rate [Hz]; integer

        Returns:
            DF - Pandas Data Frame

        Example of usage :
            df_spi = spi_TXT_to_df(path, filename, tag_arr, spi_param_arr, spi_sample_rate)
    """
    print('--> Reading SPI Data txt...')
    tsss = time.time()
    Ts = 1 / spi_sample_rate
    Fs = spi_sample_rate
    df_spi = pd.DataFrame(columns=['Time'])
    df = pd.read_csv(inputFile, header=0)
    i = 0
    for col in df.columns:
        if 'Unnamed' in col:
            # print(col)
            del df[col]
        else:
            df = df.rename(columns={col: tag_arr[i]})
            i = i + 1

    df = df.add_prefix(tag_arr[0] + '_')
    df_spi['Time'] = (df.index) * Ts
    # for col in df.columns:
    #     if 'V' in col:
    #         V_quantization=1/(2**6)
    #         df_spi[col]=df*V_quantization
    #     if 'I' in col:
    #         V_quantization=1/(2**9)
    #         df_spi[col]=df*V_quantization
    #     else:
    #         V_quantization=1/(2**12)
    #         df_spi[col]=df*V_quantization

    V_quantization = 1 / (2 ** 6)
    df_spi[df.columns] = df * V_quantization

    df_len = len(df_spi)
    df_time_len = max(df_spi['Time']) - min(df_spi['Time'])
    tmin = min(df_spi['Time'])
    tmax = max(df_spi['Time'])

    temp1 = 'DF Tmin = ' + str(tmin) + '[Sec]; ' + 'DF Tmax = ' + str(tmax) + '[Sec]; \n'
    temp2 = 'DF time length = ' + str(round(df_time_len, 5)) + '[Sec] / ~' + str(
        round(df_time_len / 60, 4)) + '[Min]; \n'
    text = temp1 + temp2 + 'DF length = ' + str(df_len / 1000000) + '[Mega Samples];\n' + 'DF Sampling rate = ' + str(
        round((Fs / 1000), 0)) + '[kSamp/sec]' + '; DF Sampling Interval = ' + str(round((Ts * 1000), 3)) + '[mSec]'

    teee = time.time()

    return (df_spi)


def sliding_spectrum_OLD(dfx, t, f, win_size, win_overlap, meas_sig):
    """
    Function performing sliding MaxHold and AVG functios on Spectrogram and returns data frame of spectrum frames

    `Z`             - Results matrix
    `t`             - time vector      (series)
    `f`             - frequency vector (series)
    `win_size`      - number of samples to calculate Max hold and AVG (integer)
    `win_overlap`   - number of samples to overlap windows (integer)
    `meas_sig`      - presented signal name (string)

    Example of usage:
        (df_MH, df_AVG, t_list)=igf.sliding_spectrum(Z, t, f, win_size, win_overlap, meas_sig)
    """

    start = time.time()
    if win_size > len(t) - 1:
        print("win size is to long! Please reduce the number of samples to be below " + str(len(t)) + " value")
        return ()

    if win_size < win_overlap:
        print(
            "win overlap size is to long! Please reduce the number of samples to be below " + str(win_size) + " value")
        return ()

    df_MH = pd.DataFrame()
    # df_AVG = pd.DataFrame()

    # df_MH = pd.DataFrame(columns=['f'])
    # df_AVG = pd.DataFrame(columns=['f'])
    # df_MH['f']=f
    # df_AVG['f']=f

    win_size = int(win_size)
    win_overlap = int(win_overlap)

    # tstep=t[1]-t[0]
    i1 = 0
    i2 = win_size
    di = win_size - win_overlap
    t_list = t[i2::di]

    while i2 < len(t):
        Zt = Z[:, i1:i2]
        Zt_AVG = Zt.mean(axis=1)
        Zt_MH = Zt.max(axis=1)
        # Zt_MH=Zt.std(axis=1)
        t1 = t[i1]
        t2 = t[i2]
        col_name = meas_sig + ' @ t = ' + str(round(t1, 3)) + '-' + str(round(t2, 3)) + ' [S]'
        df_MH[col_name] = Zt_MH
        df_AVG[col_name] = Zt_AVG
        i1 = i1 + di
        i2 = i2 + di
    end = time.time()
    return df_MH, df_AVG, t_list


def plot_all(log_energy, window_size, filter_size, over_th_limit):
    energy_th_increment = 1
    energy_th_list = [0 for x in range(window_size + filter_size)]
    sample_index = window_size + filter_size
    while sample_index < len(log_energy):
        min_filter_window = min(log_energy[(sample_index - window_size - filter_size):sample_index - window_size])
        energy_th = 0
        highest_th_is_reached = False
        while not highest_th_is_reached:
            i = sample_index - window_size
            over_threshold_counter = 0
            while i < sample_index:
                if log_energy[i] > min_filter_window + energy_th:
                    over_threshold_counter += 1
                if over_threshold_counter >= over_th_limit:
                    energy_th = energy_th + energy_th_increment
                    break
                if i == sample_index - 1:
                    highest_th_is_reached = True
                    break
                i += 1
        energy_th_list.append(energy_th)
        sample_index += 1
    return max(energy_th_list)


threadLock = threading.Lock()  # define the mutex


def energy_rise(mixer, directory, file, freq, log_energy):
    energy_th_increment = 1
    window_size = 20
    filter_size = 15
    over_th_limit = 12
    energy_th_list = [0 for x in range(window_size + filter_size)]
    sample_index = window_size + filter_size
    while sample_index < len(log_energy):
        min_filter_window = min(log_energy[(sample_index - window_size - filter_size):sample_index - window_size])
        energy_th = 0
        highest_th_is_reached = False
        while not highest_th_is_reached:
            i = sample_index - window_size
            over_threshold_counter = 0
            while i < sample_index:
                if log_energy[i] > min_filter_window + energy_th:
                    over_threshold_counter += 1
                if over_threshold_counter >= over_th_limit:
                    energy_th = energy_th + energy_th_increment
                    break
                if i == sample_index - 1:
                    highest_th_is_reached = True
                    break
                i += 1
        energy_th_list.append(energy_th)
        sample_index += 1
    threadLock.acquire()
    print(f'{mixer},{directory},{file},{freq},{max(energy_th_list)}')
    threadLock.release()


class Goertzel:
    """Contains static methods for Goertzel calculations and represents a
    "classic" Goertzel filter.
    """

    @staticmethod
    def kernel(samples, koef, v_n1, v_n2):
        """The "kernel" of the Goertzel recursive calculation.  Processes
        `samples` array of samples to pass through the filter, using the
        `k` Goertzel coefficient and the previous (two) results -
        `v_n1` and `v_n2`.  Returns the two new results.
        """

        for samp in samples:
            v_n1, v_n2 = koef*v_n1 - v_n2 + samp, v_n1
        return v_n1, v_n2

    @staticmethod
    def VOLT(koef, v_n1, v_n2, nsamp):
        """Calculates (and returns) the 'dBm', or 'dBmW' - decibel-milliwatts,
        a power ratio in dB (decibels) of the (given) measured power
        referenced to one (1) milliwat (mW).
        This uses the audio/telephony usual 600 Ohm impedance.-> i used 50 not 600
        """
        amp_x = v_n1**2 + v_n2**2 - koef*v_n1*v_n2
        if amp_x < EPSILON:
            amp_x = EPSILON
        # return 10 * np.log10(2 * amp_x * 1000 / (50*nsamp**2))

        return amp_x
    def dbm(koef, v_n1, v_n2, nsamp):
        """Calculates (and returns) the 'dBm', or 'dBmW' - decibel-milliwatts,
        a power ratio in dB (decibels) of the (given) measured power
        referenced to one (1) milliwat (mW).
        This uses the audio/telephony usual 600 Ohm impedance.-> i used 50 not 600
        """
        amp_x = v_n1**2 + v_n2**2 - koef*v_n1*v_n2
        if amp_x < EPSILON:
            amp_x = EPSILON
        return 10 * np.log10(2 * amp_x * 1000 / (50*nsamp**2))

        return amp_x
    @staticmethod

    def proc_samples_k(samples, koef):
        """Processe the given `samples` with the given `koef` Goertzel
        coefficient, returning the dBm of the signal (represented, in full,
        by the `samples`).
        """
        v_n1, v_n2 = Goertzel.kernel(samples, koef, 0, 0)
        return Goertzel.dbm(koef, v_n1, v_n2, len(samples))

    @staticmethod
    def calc_koef(freq, fsamp):
        """Calculates the Goertzel coefficient for the given frequency of the
        filter and the sampling frequency.
        """
        return 2 * math.cos(2 * math.pi * freq / fsamp)

    @staticmethod
    def process_samples(samples, freq, fsamp):
        """Processe the given +samples+ with the given Goertzel filter
        frequency `freq` and sample frequency `fsamp`, returning the
        dBm of the signal (represented, in full, by the `samples`).
        """
        return Goertzel.proc_samples_k(samples, Goertzel.calc_koef(freq, fsamp))

    def __init__(self, freq, fsamp):
        """To construct, give the frequency of the filter and the sampling
        frequency
        """
        if freq >= fsamp / 2:
            raise Exception("f is too big")
        self.freq, self.fsamp = freq, fsamp
        self.koef = Goertzel.calc_koef(freq, fsamp)
        self.zprev1 = self.zprev2 = 0

    def reset(self):
        """Reset for a new calculation"""
        self.zprev1 = self.zprev2 = 0

    def process(self, smp):
        """Process the given array of samples, return dBm"""
        self.zprev1, self.zprev2 = Goertzel.kernel(smp, self.koef, self.zprev1, self.zprev2)
        return Goertzel.dbm(self.koef, self.zprev1, self.zprev2, len(smp))

class GoertzelSampleBySample:
    """Helper class to do Goertzel algorithm sample by sample"""

    def __init__(self, freq, fsamp, nsamp,alpha,window_bool):
        """I need Frequency, sampling frequency and the number of
        samples that we shall process"""
        self.freq, self.fsamp, self.nsamp = freq, fsamp, nsamp
        self.koef = 2 * math.cos(2 * math.pi * freq / fsamp)
        self.cnt_samples = 0
        self.zprev1 = self.zprev2 = 0
        self.Erg=0
        self.ErgFiltered=0
        self.alpha = alpha
        self.Enable_prints = False
        self.enable_window=window_bool
        self.window = np.hamming(nsamp)
    def process_sample(self, samp):
        """Do one sample. Returns dBm of the input if this is the final
        sample, or None otherwise."""
        if self.enable_window:
            samp = samp * self.window[self.cnt_samples]
        Z=self.koef*self.zprev1 - self.zprev2+ (samp)
        self.zprev2 = self.zprev1
        self.zprev1=Z
        self.cnt_samples += 1
        if self.cnt_samples == self.nsamp:
            self.cnt_samples = 0
            self.Erg=Goertzel.VOLT(self.koef, self.zprev1, self.zprev2, self.nsamp)
            self.ErgFiltered=(self.alpha*self.Erg +(1-self.alpha)*self.ErgFiltered)/self.cnt_samples
            return 1
        return None
    def reset(self):
        """Reset for a new calculation"""
        self.zprev1 = 0
        self.zprev2 = 0

class GoertzelSampleBySampleFixedpoint:
    """Helper class to do Goertzel algorithm sample by sample"""
    def __init__(self, freq, fsamp, nsamp,alpha):
        """I need Frequency, sampling frequency and the number of
        samples that we shall process"""
        self.freq, self.fsamp, self.nsamp = freq, fsamp, nsamp
        self.koef = 2 * math.cos(2 * math.pi * freq / fsamp)
        self.q10koef=np.int16((1<<10)*self.koef)
        print("self.koef=" + str(self.koef))
        print("q10koef=" + str(self.q10koef))

        self.cnt_samples = np.int16(0)
        self.zprev1 = np.int16(0)
        self.zprev2 =np.int16(0)
        self.Mult=np.int32(0)
        self.Z = np.int32(0)
        self.Erg=np.int32(0)
        self.ErgFiltered=np.int32(0)
        self.alpha =int(np.log2(1/alpha));
    def process_sample(self, samp):
        """Do one sample. Returns dBm of the input if this is the final
        sample, or None otherwise."""

        # print("samp=" + str(samp))
        samp32=np.int32((np.int32(samp)-2048)<<10)
        # print("(samp-2048)<<28="+str(samp32))
        # print("zprev1=" + str(self.zprev1))
        # print("zprev2=" +str( self.zprev2))
        Mult=np.int32(np.int32(self.q10koef)*np.int32(self.zprev1))
        # print("Mult=" + str(Mult))
        Z=np.int32(np.int32(samp32)+np.int32(Mult)-(np.int32(self.zprev2)<<10))
        # print("Z=" + str(Z))
        self.zprev2= np.int16(self.zprev1)
        self.zprev1=np.int16(Z>>10)
        self.cnt_samples =  self.cnt_samples+1
        # print("cnt_samples=" + str(self.cnt_samples))

        if self.cnt_samples == self.nsamp:
            (np.int32(self.q10koef) *np.int32(self.zprev1)>>10)
            self.cnt_samples = 0
            #print("Erg=" + str(self.Erg))
            self.Erg=np.int32(self.zprev2)*np.int32(self.zprev2)  +np.int32(self.zprev1)*np.int32(self.zprev1) - ((np.int32(self.q10koef) *np.int32(self.zprev1))>>10)*np.int32(self.zprev2)
            self.ErgFiltered = np.int32(self.ErgFiltered) + np.int32((np.int32(self.Erg) - np.int32(self.ErgFiltered) >>  self.alpha));
            # exit(1)
            return 1
        #("-----------------------------------")
        return None
    def reset(self):
        """Reset for a new calculation"""
        self.zprev1 =np.int16(0)
        self.zprev2 = np.int16(0)
