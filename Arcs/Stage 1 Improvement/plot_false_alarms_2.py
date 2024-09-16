import os
import sys
import itertools
import pandas as pd
from datetime import datetime
from my_pyplot import omit_plot as _O, plot as _P, print_chrome as _PC, clear as _PP, print_lines as _PL, send_mail as _SM
import Plot_Graphs_with_Sliders as _G
import my_tools


# # exit (True) or pause (False) in case of error
exit_on_error = False
print_errors = False
# # txt output instead of the console - ATTENTION - if True, there will be no Console output:
output_text = False
path_txt = datetime.now().strftime("%d-%m-%Y %H-%M-%S")
path_txt = f'Terminal Log ({datetime.now().strftime("%d-%m-%Y %H-%M-%S")}).txt'
# # Folders and filters:
path_logs = r'M:\Users\HW Infrastructure\PLC team\ARC\Temp-Eddy\Test_False_Alarms 15_10_23 22_05'
path_output = r'M:\Users\HW Infrastructure\PLC team\ARC\Temp-Eddy'
get_only_master_slave = [False, False, 'slave']
check_if_arcs = False
Fill_NaN_with_blanks = True
Fill_blank_records = True
rec_filter_start = 0
rec_filter_end = -1
JupiterTI = False

# # PWR:
if JupiterTI:
    string_filter_pwr = 'Jupiter pwr'
    if check_if_arcs:
        search_for_pwr = 'Arc error'
        # search_for_pwr = 'Arc Detection MechState  0 --> 5'
    else:
        search_for_pwr = '; Diff = '
        # search_for_pwr = 'Arc Detection MechState  0 --> 5'
    delta_p = 0
    pwr_params = ['PWR Diff']
    sp_p = [' : ', ' = ', '_']
else:
    string_filter_pwr = 'SEDSP pwr'
    if check_if_arcs:
        search_for_pwr = 'W=<186>'
    else:
        search_for_pwr = '; Diff = '
    delta_p = 0
    pwr_params = ['PWR Diff']
    sp_p = [' : ', ' = ', '_']

# # MNGR:
if JupiterTI:
    string_filter_mngr = 'Jupiter mngr'
    if check_if_arcs:
        search_for_mngr = 'Event [186]'
    else:
        search_for_mngr = 'Event [15]'
    delta_m = 0
    mngr_params = ['MNGR Diff', 'Bitmap', 'Phase', 'Amp abs', 'Amp ratio']
    sp_m = [']   ', '$', ' ']
else:
    string_filter_mngr = 'SEDSP mngr'
    if check_if_arcs:
        search_for_mngr = '[186] ARC_PWR_DETECT'
    else:
        search_for_mngr = '[15] Arc stage2'
    delta_m = 5
    mngr_params = ['MNGR Diff', 'Bitmap', 'Phase', 'Amp abs', 'Amp ratio']
    sp_m = [':', ' ']


def main():
    arc_events = []
    if not os.path.exists(path_output):
        os.makedirs(path_output)
    if get_only_master_slave[0]:
        file_names_pwr = [f for f in os.listdir(path_logs) if f.endswith('.log') and string_filter_pwr in f and get_only_master_slave[2] not in f]
        file_names_mngr = [f for f in os.listdir(path_logs) if f.endswith('.log') and string_filter_mngr in f and get_only_master_slave[2] not in f]
    elif get_only_master_slave[1]:
        file_names_pwr = [f for f in os.listdir(path_logs) if f.endswith('.log') and string_filter_pwr in f and get_only_master_slave[2] in f]
        file_names_mngr = [f for f in os.listdir(path_logs) if f.endswith('.log') and string_filter_mngr in f and get_only_master_slave[2] in f]
    else:
        file_names_pwr = [f for f in os.listdir(path_logs) if f.endswith('.log') and string_filter_pwr in f]
        file_names_mngr = [f for f in os.listdir(path_logs) if f.endswith('.log') and string_filter_mngr in f]
    if len(file_names_pwr) != len(file_names_mngr):
        print(f'len(file_names_pwr) = {len(file_names_pwr)} != len(file_names_mngr) = {len(file_names_mngr)}!!!')
        if exit_on_error:
            exit()
        else:
            os.system("pause")
    file_names_pwr.sort()
    file_names_mngr.sort()
    if output_text:
        default_stdout = sys.stdout
        sys.stdout = open(f'{path_output}/{path_txt}', 'w')
    for record_number, (log_pwr, log_mngr) in enumerate(zip(file_names_pwr, file_names_mngr)):
        file_name = {'Rec': [match for match in log_pwr.split(' ') if 'Rec' in match][0].split('.')[0]}
        print(file_name)
        pwr_events = []
        mngr_events = []
        if rec_filter_start != 0 and record_number < rec_filter_start - 1:
            continue
        with open(f'{path_logs}\\{log_pwr}') as file:
            log = file.read().splitlines()
            for index_line, line in enumerate(log):
                if search_for_pwr in line:
                    if check_if_arcs:
                        print(f'ARC DETECTED!!! {file_name}, PWR line = {index_line + 1}')
                    else:
                        try:
                            pwr_events.append(dict(itertools.zip_longest(pwr_params, [float(l) for l in line.split(sp_p[0])[-1].split(sp_p[1])[-1].split(sp_p[2])])))
                        except:
                            if print_errors:
                                print(log_pwr + " - line " + str(index_line + delta_m) + ": " + log[index_line + delta_m])
                            pwr_events.append(dict(itertools.zip_longest(pwr_params, [float(l) for l in line.split(sp_p[0])[-1][:5].split(sp_p[1])[-1].split(sp_p[2])])))
        with open(f'{path_logs}\\{log_mngr}') as file:
            log = file.read().splitlines()
            for index_line, line in enumerate(log):
                if search_for_mngr in line:
                    if check_if_arcs:
                        print(f'ARC DETECTED!!! {file_name}, MNGR line = {index_line + 1}')
                    else:
                        try:
                            if JupiterTI:
                                mngr_events.append(dict(itertools.zip_longest(mngr_params, [float(l) for l in log[index_line + delta_m].split(']   ')[-1].split(sp_m[1])[0].split(sp_m[2])])))
                            else:
                                mngr_events.append(dict(itertools.zip_longest(mngr_params, [float(l) for l in log[index_line + delta_m].split(sp_m[0])[-1].split(sp_m[1])])))
                        except:
                            if print_errors:
                                print(log_mngr + " - line " + str(index_line + delta_m) + ": " + log[index_line + delta_m])
                            try:
                                mngr_events.append(dict(itertools.zip_longest(mngr_params, [float(l) for l in log[index_line + delta_m].split(sp_m[0])[-1][:6].split(sp_m[1])])))
                            except:
                                ...
        if not check_if_arcs:
            if print_errors and len(pwr_events) != len(mngr_events):
                print(f'ERROR!!! Rec = {file_name} (loop = {record_number})')
                print(f'len(pwr_events) = {len(pwr_events)} != len(mngr_events) = {len(mngr_events)}!!!')
            for event_number, (pwr_event, mngr_event) in enumerate(itertools.zip_longest(pwr_events, mngr_events, fillvalue={})):
                arc_events.append({**file_name, **pwr_event, **mngr_event})
        if Fill_blank_records and len(pwr_events) == 0 and len(mngr_events) == 0:
            arc_events.append(file_name)
        if rec_filter_end > 0 and record_number == rec_filter_end - 1:
            break
    if not check_if_arcs:
        df = pd.DataFrame(arc_events)
        df['Contains NaN'] = 'False'
        for index_l, line in enumerate(df.iterrows()):
            if pd.isnull(line[1].values[1]):
                if pd.isnull(line[1].values[2]):
                    df.iloc[index_l] = df.iloc[index_l].replace('False', 'No events')
                else:
                    df.iloc[index_l] = df.iloc[index_l].replace('False', 'PWR')
            elif any(pd.isnull(x) for x in line[1].values[2:-1]):
                df.iloc[index_l] = df.iloc[index_l].replace('False', 'MNGR')
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
