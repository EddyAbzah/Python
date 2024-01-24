import os
import glob
import math
import pandas as pd
import numpy as np
from io import StringIO
from my_pyplot import omit_plot as _O, plot as _P, print_chrome as _PC, clear as _PP, print_lines as _PL, send_mail as _SM
import Plot_Graphs_with_Sliders as _G
import my_tools


path_logs_string_filter_spi = 'spi'
log_delimitation_type_spi = ','
path_logs_string_filter_scope = 'scope'
log_delimitation_type_scope = ','
# ## print method calls "Getting the ..."
print_method_calls = True


def change_sync_voltage_detection(change_to):
    if print_method_calls:
        print(f'sync_voltage_detection changed to = {change_to}')


def get_files(folder_path, string_filter, spi_log_column='', file_name=''):
    """Gets all files with a specific filter: string_filter"""
    log_spi_all = []
    log_spi_names = []
    log_scope_all = []
    log_scope_names = []
    list_of_files = glob.glob(f'{folder_path}\\*{string_filter}*')
    for file_number, file in enumerate(list_of_files):
        if file_name != '' and file_name.lower() not in file.lower():
            continue
        else:
            if print_method_calls:
                find_string = '\\'
                print(f'Getting file number {file_number + 1}: name = {file[file.rindex(find_string) + 1:]}')
            if 'pwr' in file.lower() or 'mngr' in file.lower():
                continue
            if path_logs_string_filter_spi in file.lower():
                if spi_log_column == '':
                    log_spi_all.append(pd.read_csv(file, sep=log_delimitation_type_spi).dropna(axis=1))
                else:
                    log_spi_all.append(pd.read_csv(file, sep=log_delimitation_type_spi).dropna(axis=1)[spi_log_column])
                log_spi_names.append(os.path.basename(file))
            if path_logs_string_filter_scope in file.lower():
                log_scope_all.append(pd.read_csv(file, sep=log_delimitation_type_scope).dropna(axis=1))
                log_scope_names.append(os.path.basename(file))
    return log_spi_all, log_spi_names, log_scope_all, log_scope_names
