import os
import sys
import itertools
import pandas as pd
from datetime import datetime
from my_pyplot import plot as _P, print_chrome as _PC, clear as _PP, print_lines as _PL, send_mail as _SM
import Plot_Graphs_with_Sliders as _G
import my_tools


# # exit (True) or pause (False) in case of error
exit_on_error = False
# # print error or skip
skip_on_error = True
# # txt output instead of the console - ATTENTION - if True, there will be no Console output:
output_text = False
path_txt = f'Terminal Log ({datetime.now().strftime("%d-%m-%Y %H-%M-%S")}).txt'
# # Folders and filters:
path_output = r'M:\Users\HW Infrastructure\PLC team\ARC\Temp-Eddy'
path_logs = r'M:\Users\HW Infrastructure\PLC team\ARC\Temp-Eddy\Test_False_Alarms_with_MPPT 17_07_22 21_45'
check_if_arcs = False
events_to_dict = True
rec_filter = [True, 21, 40]  # bool, start, stop
inverter_v3_Jup1288 = 1      # 0 = Venus3, 1 = Jupiter 1288

# #### PWR #### #:
string_filter_pwr = 'pwr'
if check_if_arcs:
    search_for_pwr = ['ARC_DETECT_DETECTED', 'ARC_DETECT_DETECTED'][inverter_v3_Jup1288]
else:
    search_for_pwr = ['PLL SYNCED', 'Before'][inverter_v3_Jup1288]
delta_p = 0
pwr_params = ['PWR Diff']
sp_p = [' : ', ' = ', '_']

# #### MNGR #### #:
string_filter_mngr = 'mngr'
if check_if_arcs:
    search_for_mngr = ['[256]', '[186]'][inverter_v3_Jup1288]
else:
    search_for_mngr = ['<207>', '[15]'][inverter_v3_Jup1288]
delta_m = 5
mngr_params = ['MNGR Diff', 'Bitmap', 'Phase', 'Amp abs', 'Amp ratio']
sp_m = [':', ' ']


def main():
    arc_events = []
    if events_to_dict:
        all_events = {"PWR Arc": 0, "PWR Stage1": 0, "MNGR Arc": 0, "MNGR Stage1": 0}
    if output_text and not os.path.exists(path_output):
        os.makedirs(path_output)
    file_names_pwr = [f for f in os.listdir(path_logs) if f.endswith('.log') and string_filter_pwr in f]
    file_names_mngr = [f for f in os.listdir(path_logs) if f.endswith('.log') and string_filter_mngr in f]
    print(f'len(file_names_pwr) = {len(file_names_pwr)} _ len(file_names_mngr) = {len(file_names_mngr)}')
    if len(file_names_pwr) != len(file_names_mngr):
        print(f'len(file_names_pwr) = {len(file_names_pwr)} != len(file_names_mngr) = {len(file_names_mngr)}!!!')
        if exit_on_error:
            exit()
        else:
            os.system("pause")
    file_names_pwr = sorted(file_names_pwr)
    file_names_mngr = sorted(file_names_mngr)
    if output_text:
        default_stdout = sys.stdout
        sys.stdout = open(f'{path_output}/{path_txt}', 'w')
    for record_number, (log_pwr, log_mngr) in enumerate(zip(file_names_pwr, file_names_mngr)):
        if rec_filter[0] and (record_number < rec_filter[1] - 1 or record_number >= rec_filter[2]):
            continue
        file_name = {'Rec': [match for match in log_pwr.split(' ') if 'Rec' in match][0].split('.')[0]}
        if events_to_dict:
            pwr_events = {"Arc": 0, "Stage1": 0}
            mngr_events = {"Arc": 0, "Stage1": 0}
        else:
            pwr_events = []
            mngr_events = []
        with open(f'{path_logs}\\{log_pwr}') as file:
            log = file.read().splitlines()
            for index_line, line in enumerate(log):
                if search_for_pwr in line:
                    if check_if_arcs:
                        if events_to_dict:
                            pwr_events["Arc"] = pwr_events["Arc"] + 1
                        else:
                            print(f'ARC DETECTED!!! {file_name}, PWR line = {index_line + 1}')
                    else:
                        if events_to_dict:
                            pwr_events["Stage1"] = pwr_events["Stage1"] + 1
                        else:
                            pwr_events.append(dict(itertools.zip_longest(pwr_params, [float(l) for l in line.split(sp_p[0])[-1].split(sp_p[1])[-1].split(sp_p[2])])))
        with open(f'{path_logs}\\{log_mngr}') as file:
            log = file.read().splitlines()
            for index_line, line in enumerate(log):
                if search_for_mngr in line:
                    if check_if_arcs:
                        if events_to_dict:
                            mngr_events["Arc"] = mngr_events["Arc"] + 1
                        else:
                            print(f'ARC DETECTED!!! {file_name}, MNGR line = {index_line + 1}')
                    else:
                        if events_to_dict:
                            mngr_events["Stage1"] = mngr_events["Stage1"] + 1
                        else:
                            mngr_events.append(dict(itertools.zip_longest(mngr_params, [float(l) for l in log[index_line + delta_m].split(sp_m[0])[-1].split(sp_m[1])])))
        if not check_if_arcs:
            if skip_on_error:
                print(f'{file_name.__str__()[9:15]}\t{pwr_events["Stage1"]}\t{mngr_events["Stage1"]}')
            else:
                if (events_to_dict and pwr_events["Stage1"] != mngr_events["Stage1"]) or (not events_to_dict and len(pwr_events) != len(mngr_events)):
                    print(f'ERROR!!! Rec = {file_name} (loop = {record_number})')
                    if events_to_dict:
                        print(f'pwr_events["Stage1"] = {pwr_events["Stage1"]} != mngr_events["Stage1"] = {mngr_events["Stage1"]}!!!')
                    else:
                        print(f'len(pwr_events) = {len(pwr_events)} != len(mngr_events) = {len(mngr_events)}!!!')
                if not events_to_dict:
                    for event_number, (pwr_event, mngr_event) in enumerate(itertools.zip_longest(pwr_events, mngr_events, fillvalue={})):
                        arc_events.append({**file_name, **pwr_event, **mngr_event})
        if events_to_dict:
            all_events["PWR Arc"] = all_events["PWR Arc"] + pwr_events["Arc"]
            all_events["PWR Stage1"] = all_events["PWR Stage1"] + pwr_events["Stage1"]
            all_events["MNGR Arc"] = all_events["MNGR Arc"] + mngr_events["Arc"]
            all_events["MNGR Stage1"] = all_events["MNGR Stage1"] + mngr_events["Stage1"]
    if events_to_dict:
        print(f"PWR Arc events = " + str(all_events["PWR Arc"]))
        print(f"PWR Stage1 events = " + str(all_events["PWR Stage1"]))
        print(f"MNGR Arc events = " + str(all_events["MNGR Arc"]))
        print(f"MNGR Stage1 events = " + str(all_events["MNGR Stage1"]))
    else:
        if not check_if_arcs:
            df = pd.DataFrame(arc_events)
            df['Contains NaN'] = 'False'
            for index_l, line in enumerate(df.iterrows()):
                for index_v, value in enumerate(line[1].values):
                    if pd.isnull(value):
                        if index_v <= len(pwr_params):
                            df.iloc[index_l] = df.iloc[index_l].replace('False', 'PWR')
                        else:
                            df.iloc[index_l] = df.iloc[index_l].replace('False', 'MNGR')

        print()
        print(df.to_string())
    if output_text:
        sys.stdout.close()
        sys.stdout = default_stdout


if __name__ == "__main__":
    main()
