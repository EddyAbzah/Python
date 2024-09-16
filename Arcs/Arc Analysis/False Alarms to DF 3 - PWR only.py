import os
import re
import sys
import csv
import itertools
import pandas as pd
from datetime import datetime
if os.getlogin() == "eddy.a":
    from my_pyplot import plot as _P, print_chrome as _PC, clear as _PP, print_lines as _PL, send_mail as _SM
    import Plot_Graphs_with_Sliders as _G
    import my_tools

# ####   True   ###   False   #### #
venus_T_or_jupiter_F = False
stage1_1__stage2_2__both_3 = 2
exit_on_error = False   # or pause
print_errors = True
output_text = False   # txt output instead of the console
path_txt = f'Terminal Log ({datetime.now().strftime("%d-%m-%Y %H-%M-%S")}).txt'
# ####   Folders   and   filters   #### #
path_output = r'M:\Users\HW Infrastructure\PLC team\ARC\Temp-Eddy'
path_logs = r'M:\Users\HW Infrastructure\PLC team\ARC\Temp-Noam\Test_False_Alarms 19_02_24 20_58'
check_if_arcs = False
Fill_NaN_with_blanks = True
rec_filter__start_end = [False, 101, 666]
find_info_in_folder = [False, '_RecInfo.csv', lambda group: int((group - 1) / find_info_in_folder[3]), 0]


if venus_T_or_jupiter_F:
    string_filter_pwr = 'Venus3 DSP pwr'
    if check_if_arcs:
        search_for_pwr = '<186>'
    else:
        if stage1_1__stage2_2__both_3 == 1:
            search_for_pwr = '<207>'
            pwr_params = ['Max Current Drop', 'Power Before', 'Abs Amps Before', 'Max Energy', 'Erg Floor']
        elif stage1_1__stage2_2__both_3 == 2:
            search_for_pwr = '<15>'
            pwr_params = ['Power Diff', 'Bitmap', 'Phase', 'Amp abs', 'Amp ratio']
        else:
            search_for_pwr_arr = ['<207>', '<15>']
            pwr_params_arr = [['Max Current Drop', 'Power Before', 'Abs Amps Before', 'Max Energy', 'Erg Floor'], ['Power Diff', 'Bitmap', 'Phase', 'Amp abs', 'Amp ratio']]
else:
    string_filter_pwr = 'SEDSP pwr'
    if check_if_arcs:
        search_for_pwr = 'STAGE2 DETECT: TRUE'
    else:
        if stage1_1__stage2_2__both_3 == 1:
            search_for_pwr = 'Ev166 Struct:'
            pwr_params = ['Energy Rise', 'Current Drop', 'AAA', 'BBB', 'CCC']
        elif stage1_1__stage2_2__both_3 == 2:
            search_for_pwr = 'Ev15 Struct:'
            pwr_params = ['Power Diff', 'Bitmap', 'Phase', 'Amp abs', 'Amp ratio']
        else:
            search_for_pwr_arr = ['Ev166 Struct:', 'Ev15 Struct:']
            pwr_params_arr = [['Energy Rise', 'Current Drop', 'AAA', 'BBB', 'CCC'], ['Power Diff', 'Bitmap', 'Phase', 'Amp abs', 'Amp ratio']]


def main():
    global search_for_pwr
    global pwr_params
    if not os.path.exists(path_output):
        os.makedirs(path_output)
    file_names_pwr = [f for f in os.listdir(path_logs) if f.endswith('.log') and string_filter_pwr in f]
    file_names_pwr.sort()
    pwr_events = []
    if find_info_in_folder[0]:
        with open(path_logs + '\\' + find_info_in_folder[1], 'r') as f:
            f = f.readlines()
            find_info_in_folder[3] = int(f[0].split('=')[-1].split('\n')[0])
            rec_info = list(csv.DictReader(f[1:]))
    if output_text:
        default_stdout = sys.stdout
        sys.stdout = open(f'{path_output}/{path_txt}', 'w')
    for record_number, log_pwr in enumerate(file_names_pwr):
        try:
            record_number = int(log_pwr[log_pwr.lower().find('rec') + 3:log_pwr.lower().find('rec') + 6])
        except:
            pass
        if rec_filter__start_end[0] and record_number < rec_filter__start_end[1]:
            continue
        if rec_filter__start_end[0] and record_number > rec_filter__start_end[2]:
            break
        file_name = {'Rec': [match for match in log_pwr.split(' ') if 'Rec' in match][0].split('.')[0]}

        with open(f'{path_logs}\\{log_pwr}') as file:
            log = file.read().splitlines()
            stage1_exists = False
            for index_line, line in enumerate(log):
                if stage1_1__stage2_2__both_3 == 3:
                    if not stage1_exists:
                        search_for_pwr = search_for_pwr_arr[0]
                        pwr_params = pwr_params_arr[0]
                    else:
                        search_for_pwr = search_for_pwr_arr[1]
                        pwr_params = pwr_params_arr[1]
                if search_for_pwr in line:
                    if check_if_arcs:
                        print(f'ARC DETECTED!!! {file_name}, PWR line = {index_line + 1}')
                    else:
                        try:
                            if not any(s in line.split(search_for_pwr)[-1] for s in ['<', '>']):
                                if stage1_1__stage2_2__both_3 != 3 or not stage1_exists:
                                    pwr_events.append(dict(itertools.zip_longest(pwr_params, re.findall(r'-?\d+\.\d+', line.split(search_for_pwr)[-1]))))
                                    pwr_events[-1].update(file_name)
                                    if find_info_in_folder[0]:
                                        pwr_events[-1].update(rec_info[find_info_in_folder[2](record_number)])
                                else:
                                    pwr_events[-1].update(dict(itertools.zip_longest(pwr_params, re.findall(r'-?\d+\.\d+', line.split(search_for_pwr)[-1]))))
                                stage1_exists = not stage1_exists
                        except:
                            if print_errors:
                                print(f'{log_pwr} - line {index_line}: {log[index_line]}')
            if stage1_1__stage2_2__both_3 == 3 and stage1_exists:
                pwr_events.pop()
    if not check_if_arcs:
        df = pd.DataFrame(pwr_events)
        df['Contains NaN'] = 'False'
        for index_l, line in enumerate(df.iterrows()):
            if pd.isnull(line[1].values[1]):
                df.iloc[index_l] = df.iloc[index_l].replace('False', 'True')
        if Fill_NaN_with_blanks:
            df = df.fillna('')
        print()
        print("\t".join(map(str, list(df.columns.values))))
        for i in range(len(df.index)):
            row = list(df.iloc[i])
            print("\t".join(map(str, row)))
    if output_text:
        sys.stdout.close()
        sys.stdout = default_stdout


if __name__ == "__main__":
    main()
