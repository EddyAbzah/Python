import numpy as np
import scipy.signal
import heapq
from my_pyplot import omit_plot as _O, plot as _P, print_chrome as _PC, clear as _PP, print_lines as _PL, send_mail as _SM
import Plot_Graphs_with_Sliders as _G
import my_tools


# ## put True to initialize threshold lists with = [0 for x in range(window_size + filter_size)]
initialize_empty_lists = False
# ## same lists as above:
sync_energy_detection = -12  # this is better
sync_energy_detection = 0
sync_current_detection = -13  # this is better
sync_current_detection = 0
# ## set this in relation to Voltage Drop parameters (in "main.py")
sync_voltage_detection = 0
# ## "th_increment" (both for Energy Rise and Current Drop) → lower numbers = higher definition BUT longer process time:
energy_th_increment_1 = 1
energy_th_increment_2 = 1
current_th_increment_1 = 0.02
current_th_increment_2 = 0.1
current_th_increment_3 = 1
voltage_th_increment = 0.01
# ## for "scipy.signal.find_peaks()" more information at: docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html
prominence = 9
height = 10
threshold = None
# ## print method calls "Getting the ..."
print_method_calls = False


def change_sync_voltage_detection(change_to):
    global sync_voltage_detection
    sync_voltage_detection = change_to
    print(f'sync_voltage_detection changed to = {sync_voltage_detection}')


def plot_all(log_energy, window_size, filter_size, over_th_limit):
    if print_method_calls:
        print(f'Getting the Energy Rise; initialize_empty_lists = {initialize_empty_lists}')
    if initialize_empty_lists:
        energy_th_list = []
    else:
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
                    if energy_th < 20:
                        energy_th = energy_th + energy_th_increment_1
                    else:
                        energy_th = energy_th + energy_th_increment_2
                    break
                if i == sample_index - 1:
                    highest_th_is_reached = True
                    break
                i += 1
        energy_th_list.append(energy_th)
        sample_index += 1
    if sync_energy_detection != 0:
        if sync_energy_detection > 0:
            energy_th_list[0:0] = [0 for i in range(sync_energy_detection)]
            del energy_th_list[-sync_energy_detection:]
        else:
            sync_energy_detection_absolute = abs(sync_energy_detection)
            del energy_th_list[:sync_energy_detection_absolute]
            energy_th_list.extend([0 for i in range(sync_energy_detection_absolute)])
    return energy_th_list


def plot_all_Vout(log_energy, window_size, filter_size, over_th_limit):
    if print_method_calls:
        print(f'Getting the Energy Rise')
    if initialize_empty_lists:
        energy_th_list = []
    else:
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
                    if energy_th < 20:
                        energy_th = energy_th + energy_th_increment_1
                    else:
                        energy_th = energy_th + energy_th_increment_2
                    break
                if i == sample_index - 1:
                    highest_th_is_reached = True
                    break
                i += 1
        energy_th_list.append(energy_th)
        sample_index += 1
    if sync_energy_detection != 0:
        if sync_energy_detection > 0:
            energy_th_list[0:0] = [0 for i in range(sync_energy_detection)]
            del energy_th_list[-sync_energy_detection:]
        else:
            sync_energy_detection_absolute = abs(sync_energy_detection)
            del energy_th_list[:sync_energy_detection_absolute]
            energy_th_list.extend([0 for i in range(sync_energy_detection_absolute)])
    return energy_th_list


def current_algo_old(log_current, window_size, filter_size, over_th_limit):
    if print_method_calls:
        print(f'Getting the Current Drop (old algorithm); initialize_empty_lists = {initialize_empty_lists}')
    if initialize_empty_lists:
        current_th_list = []
    else:
        current_th_list = [0 for x in range(window_size + filter_size)]
    sample_index = window_size + filter_size
    while sample_index < len(log_current):
        avg_filter_window = (log_current[(sample_index - window_size - filter_size):sample_index - window_size]).mean()
        current_th = 0.0
        highest_th_is_reached = False
        while not highest_th_is_reached:
            i = sample_index - window_size
            over_threshold_counter = 0
            while i < sample_index:
                if avg_filter_window - log_current[i] > current_th:
                    over_threshold_counter += 1
                if over_threshold_counter >= over_th_limit:
                    if current_th < 1:
                        current_th = current_th + current_th_increment_1
                    elif current_th < 5:
                        current_th = current_th + current_th_increment_2
                    else:
                        current_th = current_th + current_th_increment_3
                    break
                if i == sample_index - 1:
                    highest_th_is_reached = True
                    break
                i += 1
        current_th_list.append(current_th)
        sample_index += 1
    if sync_current_detection != 0:
        if sync_current_detection > 0:
            current_th_list[0:0] = [0 for i in range(sync_current_detection)]
            del current_th_list[-sync_current_detection:]
        else:
            sync_current_detection_absolute = abs(sync_current_detection)
            del current_th_list[:sync_current_detection_absolute]
            current_th_list.extend([0 for i in range(sync_current_detection_absolute)])
    return current_th_list


def current_algo_new_older_format(log_current, window_size, filter_size, over_th_limit, energy_drop_th):
    if print_method_calls:
        print(f'Getting the Current Drop (new algorithm but older format); initialize_empty_lists = {initialize_empty_lists}')
    if initialize_empty_lists:
        current_th_list = []
    else:
        current_th_list = [0 for x in range(window_size + filter_size)]
    ac_voltage = 230
    sample_rate = 1 / 35
    sample_index = window_size + filter_size
    while sample_index < len(log_current):
        avg_filter_window = (log_current[(sample_index - window_size - filter_size):sample_index - window_size]).mean()
        current_th = 0.0
        highest_th_is_reached = False
        while not highest_th_is_reached:
            i = sample_index - window_size
            over_threshold_counter = 0
            energy_drop = 0
            while i < sample_index:
                if avg_filter_window - log_current[i] > current_th:
                    over_threshold_counter += 1
                    di = avg_filter_window - log_current[i]
                    energy_drop += di * sample_rate * ac_voltage
                    if energy_drop >= energy_drop_th and over_threshold_counter >= over_th_limit:
                        if current_th < 1:
                            current_th = current_th + current_th_increment_1
                        elif current_th < 5:
                            current_th = current_th + current_th_increment_2
                        else:
                            current_th = current_th + current_th_increment_3
                        break
                if i == sample_index - 1:
                    highest_th_is_reached = True
                    break
                i += 1
        current_th_list.append(current_th)
        sample_index += 1
    if sync_current_detection != 0:
        if sync_current_detection > 0:
            current_th_list[0:0] = [0 for i in range(sync_current_detection)]
            del current_th_list[-sync_current_detection:]
        else:
            sync_current_detection_absolute = abs(sync_current_detection)
            del current_th_list[:sync_current_detection_absolute]
            current_th_list.extend([0 for i in range(sync_current_detection_absolute)])
    return current_th_list


def current_algo_new(log_current, window_size, filter_size, over_th_limit, energy_drop_th):
    if print_method_calls:
        print(f'Getting the Current Drop (new algorithm); initialize_empty_lists = {initialize_empty_lists}')
    if initialize_empty_lists:
        current_th_list = []
    else:
        current_th_list = [0 for x in range(window_size + filter_size)]
    ac_voltage = 230
    sample_rate = 1 / 35
    sample_index = window_size + filter_size
    while sample_index < len(log_current):
        avg_filter_window = (log_current[(sample_index - window_size - filter_size):sample_index - window_size]).mean()
        i = sample_index - window_size
        energy_drop = abs(sum((avg_filter_window - log_current[i:sample_index]) * sample_rate * ac_voltage))
        current_th = 0.0
        highest_th_is_reached = False
        while not highest_th_is_reached:
            i = sample_index - window_size
            over_threshold_counter = 0
            while i < sample_index:
                if avg_filter_window - log_current[i] > current_th:
                    over_threshold_counter += 1
                    if energy_drop >= energy_drop_th and over_threshold_counter >= over_th_limit:
                        if current_th < 1:
                            current_th = current_th + current_th_increment_1
                        elif current_th < 5:
                            current_th = current_th + current_th_increment_2
                        else:
                            current_th = current_th + current_th_increment_3
                        break
                if i == sample_index - 1:
                    highest_th_is_reached = True
                    break
                i += 1
        current_th_list.append(current_th)
        sample_index += 1
    if sync_current_detection != 0:
        if sync_current_detection > 0:
            current_th_list[0:0] = [0 for i in range(sync_current_detection)]
            del current_th_list[-sync_current_detection:]
        else:
            sync_current_detection_absolute = abs(sync_current_detection)
            del current_th_list[:sync_current_detection_absolute]
            current_th_list.extend([0 for i in range(sync_current_detection_absolute)])
    return current_th_list


def voltage_algo(log_voltage, window_size, filter_size, over_th_limit, true_if_avg):
    """Returns the TH only if the over_th_limit is reached"""
    if print_method_calls:
        print(f'Getting the Voltage Drop; initialize_empty_lists = {initialize_empty_lists}')
    if initialize_empty_lists:
        voltage_th_list = []
    else:
        voltage_th_list = [0 for x in range(window_size + filter_size)]
    sample_index = window_size + filter_size
    while sample_index < len(log_voltage):
        if true_if_avg:
            filter_window = (log_voltage[(sample_index - window_size - filter_size):sample_index - window_size]).mean()
        else:
            filter_window = min(log_voltage[(sample_index - window_size - filter_size):sample_index - window_size])
        voltage_th = 0
        highest_th_is_reached = False
        while not highest_th_is_reached:
            i = sample_index - window_size
            over_threshold_counter = 0
            while i < sample_index:
                if filter_window - log_voltage[i] > voltage_th:
                    over_threshold_counter += 1
                if over_threshold_counter >= over_th_limit:
                    voltage_th = voltage_th + voltage_th_increment
                    break
                if i == sample_index - 1:
                    highest_th_is_reached = True
                    break
                i += 1
        voltage_th_list.append(voltage_th)
        sample_index += 1
    if sync_voltage_detection != 0:
        if sync_voltage_detection > 0:
            voltage_th_list[0:0] = [0 for i in range(sync_voltage_detection)]
            del voltage_th_list[-sync_voltage_detection:]
        else:
            sync_voltage_detection_absolute = abs(sync_voltage_detection)
            del voltage_th_list[:sync_voltage_detection_absolute]
            voltage_th_list.extend([0 for i in range(sync_voltage_detection_absolute)])
    return voltage_th_list


def voltage_algo(log_voltage, window_size, filter_size, over_th_limit, true_if_avg, voltage_th):
    """Returns the TH only if the over_th_limit is reached"""
    if print_method_calls:
        print(f'Getting the Voltage Drop; initialize_empty_lists = {initialize_empty_lists}')
    if initialize_empty_lists:
        filter_window_list = []
        voltage_th_list = []
    else:
        filter_window_list = [0 for x in range(window_size + filter_size)]
        voltage_th_list = [0 for x in range(window_size + filter_size)]
    sample_index = window_size + filter_size
    while sample_index < len(log_voltage):
        if true_if_avg:
            filter_window = (log_voltage[(sample_index - window_size - filter_size):sample_index - window_size]).mean()
        else:
            filter_window = min(log_voltage[(sample_index - window_size - filter_size):sample_index - window_size])
        filter_window_list.append(filter_window)
        highest_th_is_reached = False
        while not highest_th_is_reached:
            i = sample_index - window_size
            over_threshold_counter = 0
            while i < sample_index:
                if filter_window - log_voltage[i] > voltage_th:
                    over_threshold_counter += 1
                if over_threshold_counter >= over_th_limit:
                    highest_th_is_reached = True
                    break
                if i == sample_index - 1:
                    highest_th_is_reached = True
                    break
                i += 1
        voltage_th_list.append(over_threshold_counter)
        sample_index += 1
    if sync_voltage_detection != 0:
        if sync_voltage_detection > 0:
            voltage_th_list[0:0] = [0 for i in range(sync_voltage_detection)]
            del voltage_th_list[-sync_voltage_detection:]
        else:
            sync_voltage_detection_absolute = abs(sync_voltage_detection)
            del voltage_th_list[:sync_voltage_detection_absolute]
            voltage_th_list.extend([0 for i in range(sync_voltage_detection_absolute)])
    return voltage_th_list, filter_window_list


def voltage_over_th(log_voltage, window_size, filter_size, true_if_avg):
    """Returns the TH without regard to over_th_limit"""
    if print_method_calls:
        print(f'Getting the Voltage Drop Over Thresholds; initialize_empty_lists = {initialize_empty_lists}')
    if initialize_empty_lists:
        voltage_th_list = []
    else:
        voltage_th_list = [0 for x in range(window_size + filter_size)]
    sample_index = window_size + filter_size
    while sample_index < len(log_voltage):
        if true_if_avg:
            filter_window = (log_voltage[(sample_index - window_size - filter_size):sample_index - window_size]).mean()
        else:
            filter_window = min(log_voltage[(sample_index - window_size - filter_size):sample_index - window_size])
        over_th_limit = 0
        i = sample_index - window_size
        while i < sample_index:
            if filter_window > log_voltage[i]:
                over_th_limit += filter_window - log_voltage[i]
            i += 1
        voltage_th_list.append(over_th_limit)
        sample_index += 1
    if sync_voltage_detection != 0:
        if sync_voltage_detection > 0:
            voltage_th_list[0:0] = [0 for i in range(sync_voltage_detection)]
            del voltage_th_list[-sync_voltage_detection:]
        else:
            sync_voltage_detection_absolute = abs(sync_voltage_detection)
            del voltage_th_list[:sync_voltage_detection_absolute]
            voltage_th_list.extend([0 for i in range(sync_voltage_detection_absolute)])
    return voltage_th_list


def print_ranges(index_file, file_name, cut_log_at, find_peaks, print_list, print_titles):
    print(f'Ranges {index_file:03}, def print_ranges() = {file_name}')
    print()
    if cut_log_at == 0:   # no Arc has been detected
        for index, data_frame in enumerate(print_list):
            if print_titles[index] == 'Energy [dB]':
                print(f'Ranges {index_file:03}, {print_titles[index]}, whole record, MIN = {np.min(data_frame)}')
            print(f'Ranges {index_file:03}, {print_titles[index]}, whole record, MAX = {np.max(data_frame)}')
            print(f'Ranges {index_file:03}, {print_titles[index]}, whole record, AVG = {np.mean(data_frame)}')
            if find_peaks and print_titles[index] == 'Energy [dB]':
                indexes, values = scipy.signal.find_peaks(data_frame, prominence=prominence,
                                                          height=height, threshold=threshold)
                print(f'Ranges {index_file:03}, Energy Peaks, whole record, count = {len(indexes)}')
                # to print the peaks:
                # for x_axis in indexes:
                #     print(f'{x_axis:06} = {data_frame[x_axis]}')
            print()
    else:
        for index, data_frame in enumerate(print_list):
            if print_titles[index] == 'Energy [dB]':
                print(f'Ranges {index_file:03}, {print_titles[index]}, before cut, MIN = {np.min(data_frame[0:cut_log_at])}')
            print(f'Ranges {index_file:03}, {print_titles[index]}, before cut, MAX = {np.max(data_frame[0:cut_log_at])}')
            print(f'Ranges {index_file:03}, {print_titles[index]}, before cut, AVG = {np.mean(data_frame[0:cut_log_at])}')
            if find_peaks and print_titles[index] == 'Energy [dB]':
                indexes, values = scipy.signal.find_peaks(data_frame[0:cut_log_at], prominence=prominence,
                                                          height=height, threshold=threshold)
                print(f'Ranges {index_file:03}, Energy Peaks, before cut, count = {len(indexes)}')
            if print_titles[index] == 'Energy [dB]':
                print(f'Ranges {index_file:03}, {print_titles[index]}, after cut, MIN = {np.min(data_frame[cut_log_at:])}')
            print(f'Ranges {index_file:03}, {print_titles[index]}, after cut, MAX = {np.max(data_frame[cut_log_at:])}')
            print(f'Ranges {index_file:03}, {print_titles[index]}, after cut, AVG = {np.mean(data_frame[cut_log_at:])}')
            if find_peaks and print_titles[index] == 'Energy [dB]':
                indexes, values = scipy.signal.find_peaks(data_frame[cut_log_at:], prominence=prominence,
                                                          height=height, threshold=threshold)
                print(f'Ranges {index_file:03}, Energy Peaks, after cut, count = {len(indexes)}')
            print()


def print_ranges_2(index_file, file_name, print_list, print_titles, alpha_filter):
    print()
    print(f'Ranges {index_file:03}, def print_ranges_2(), file name = {file_name}')
    for index_data_frame, data_frame in enumerate(print_list):
        print(f'Ranges {index_file:03}, {print_titles[index_data_frame]}, {print_titles[index_data_frame]}, {print_titles[index_data_frame]}, {print_titles[index_data_frame]}, {print_titles[index_data_frame]}')
        print(f'Ranges {index_file:03}, alpha, MAX, MAX next, MAX ratio, MAX delta')
        for index_trace, trace in enumerate(data_frame):
            indexes, values = scipy.signal.find_peaks(trace, prominence=0.01)
            try:
                max_values = heapq.nlargest(2, values["prominences"])
                ratio = round(max_values[0] / max_values[1], 2)
                delta = round(max_values[0] - max_values[1], 2)
                max_values = [round(num, 2) for num in max_values]
                print(f'Ranges {index_file:03}, {alpha_filter[index_trace]}, {max_values[0]}, {max_values[1]}, {ratio}, {delta}')
            except (IndexError, KeyError):
                # # The "prominences" aren't prominent enough
                print(f'Ranges {index_file:03}, {alpha_filter[index_trace]}, N/A, N/A, N/A, N/A')
        print()


def noise_floor_AVG(log_energy, Noise_floor_AVG_samples):
    if print_method_calls:
        print(f'Getting the Noise Floor AVG for {Noise_floor_AVG_samples} samples')
    # see link for info: https://stackoverflow.com/questions/13728392/moving-average-or-running-mean
    return np.convolve(log_energy, np.ones(Noise_floor_AVG_samples)/Noise_floor_AVG_samples, mode='valid')
