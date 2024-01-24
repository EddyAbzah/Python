import os
import sys
from datetime import datetime
import pandas as pd
import numpy as np
from my_pyplot import omit_plot as _O, plot as _P, print_chrome as _PC, clear as _PP, print_lines as _PL, send_mail as _SM
import Plot_Graphs_with_Sliders as _G
import my_tools


new_type = True
output_text = True
print_prints = False
# ## True ### False ## #
path_output = r'M:\Users\HW Infrastructure\PLC team\ARC\Temp-Eddy'
if new_type:
    path_folder = r'V:\HW_Infrastructure\Analog_Team\ARC_Data\Results\Venus3\11.4kW DSP (7403495A)\New Arc Detection frequency (24-10-2021)\CSVs - Rolling averages'
    path_filter = 'All tests'
else:
    path_folder = r'M:\Users\HW Infrastructure\PLC team\ARC\Temp-Eddy\Energy Rise - CSVs'
    tests = ['Noise floor No Telems', 'Force Telemetries discrete', 'False Alarms case 9']
    path_filter = 'Mixer'
tests = ['Noise floor No Telems', 'Force Telemetries discrete', 'False Alarms case 9']
pairings = ['TX1-RX1', 'TX1-RX2', 'TX1-RX3', 'TX2-RX1', 'TX2-RX2', 'TX2-RX3']


if not os.path.exists(path_output):
    os.makedirs(path_output)
if output_text:
    default_stdout = sys.stdout
    sys.stdout = open(f'{path_output}/Terminal Log ({datetime.now().strftime("%d-%m-%Y %H-%M-%S")}).txt', 'w')
file_names = [f for f in os.listdir(path_folder) if f.endswith(f'.csv') and path_filter in f]
if print_prints:
    print(f'Got files; length = {len(file_names)}')
for index_file, file in enumerate(file_names):
    if not new_type:
        mixer = file.split(' = ')[-1].split('.')[0]
        file_name = file.split(' = ')[0]
    all_prints = []
    if print_prints:
        print()
        print(f'Loop → index_file = {index_file + 1} _ file - {file}')
        print()
    df_all = pd.read_csv(f'{path_folder}\\{file}')
    for pairing in pairings:
        prints = []
        for test in tests:
            if print_prints:
                print()
                print(f'Getting numbers for {pairing} - {test}:')
                print()
            all_prints.append([f'{pairing} _ {test}', *df_all.loc[df_all['Pairing'] == pairing].loc[df_all['Test'] == test]['Max Rolling Average 250']])
        all_prints.append([f'{pairing} _ Telems Delta', *[a - b for a, b in zip(all_prints[-2][1:], all_prints[-3][1:])]])
        all_prints.append([f'{pairing} _ FAs Delta', *[a - b for a, b in zip(all_prints[-2][1:], all_prints[-4][1:])]])
        if new_type and pairing == 'TX2-RX3':
            all_prints.append([f'{pairing} _ Arcs', *df_all.loc[df_all['Pairing'] == pairing].loc[df_all['Test'] == 'Arcs']['Max Rolling Average 250']])
            all_prints.append([f'{pairing} _ Arcs Delta', *[a - b for a, b in zip(all_prints[-1][1:], all_prints[-6][1:])]])

    prints = np.array(all_prints).T
    if print_prints:
        print()
        print(f'{file}:')
        print()
    if not new_type:
        for row in prints:
            print(",".join([mixer] + [f for f in row]))
    else:
        for row in prints:
            print('\t'.join([f for f in row]))

if print_prints:
    print()
    print(f'Finito')
if output_text:
    sys.stdout.close()
    sys.stdout = default_stdout