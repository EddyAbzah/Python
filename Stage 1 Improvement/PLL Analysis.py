import math
import glob
import scipy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.signal import square, sawtooth, correlate
from my_pyplot import omit_plot as _O, plot as _P, print_chrome as _PC, clear as _PP, print_lines as _PL, send_mail as _SM
import Plot_Graphs_with_Sliders as _G
import my_tools


# ###   True   ###   False   ### #
print_diffs_only = True
print_graphs = False
export_for_PLL = True
# # Files:
path_foler = r'M:\Users\HW Infrastructure\PLC team\ARC\Temp-Eddy\Digital_Record 07_08_23 14_22'
path_file_name = ['Scope_Rec', '.csv']
path_output_file_and_title = 'Spectrum 1kHz-260kHz'
# # Output:
first_column_is_x_axis = [True, np.linspace(1e3, 260e3, num=1001)]
filters = [850, 2150, 4800, 9000]


all_files = glob.glob(path_foler + '\\*' + path_file_name[0] + '*' + path_file_name[1])
for file in all_files:
    df = pd.read_csv(file).dropna(axis=1, how='all')
    print(file.split('\\')[-1])
    df.columns = ['Time', 'Ref', 'PLL', 'Current']
    if export_for_PLL:
        df['PLL'].rename(file.split('\\')[-1][:-4]).to_csv(file[:-4] + ' PLL.txt', index=False)
    dfs = [df.iloc[filters[0]:filters[1], ], df.iloc[filters[2]:filters[3], ]]
    if print_graphs:
        plt.plot(df)
        plt.plot(dfs[0])
        plt.plot(dfs[1])
        plt.show()
    A_diffs = []
    B_diffs = []
    C_diffs = []
    for df in dfs:
        A = scipy.signal.correlate(df['Ref'], df['PLL'])
        A_diffs.append(A.mean())
        if not print_diffs_only:
            print('Method A.mean() = ' + str(A.mean()))
        if print_graphs:
            plt.plot(A)
            plt.show()

        B = np.arctan2(df['Ref'], df['PLL'])
        B_diffs.append(B.mean())
        if not print_diffs_only:
            print('Method B.mean() = ' + str(B.mean()))
        if print_graphs:
            plt.plot(B)
            plt.show()

        # Method C
        t = np.array(df['Time'])
        rdata = np.zeros((len(t), 2))
        rdata[:, 0] = np.array(df['Ref'])
        rdata[:, 1] = np.array(df['PLL'])

        # scale
        scaler = StandardScaler()
        scaler.fit(rdata)
        data = scaler.transform(rdata)

        if print_graphs:
            plt.figure(figsize=(4, 4))
            plt.title('Phase diagram')
            plt.scatter(data[:, 0], data[:, 1])
            plt.show()

        c = np.cov(np.transpose(data))
        phi = np.arccos(c[0, 1])
        C_diffs.append(phi / math.pi * 180)
        if not print_diffs_only:
            print(f'Method C cov = {c[0]} {c[1]}')
            print('Method C phase estimate (radians): ', phi, '(degrees): ', phi / math.pi * 180)
            print('')
    print(f'Method A diff = {A_diffs[1] - A_diffs[0]}')
    print(f'Method B diff = {B_diffs[1] - B_diffs[0]}')
    print(f'Method C diff = {C_diffs[1] - C_diffs[0]}')
    print('')
    print('')