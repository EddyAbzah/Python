import pandas as pd
import numpy as np
from my_pyplot import omit_plot as _O, plot as _P, print_chrome as _PC, clear as _PP, print_lines as _PL, send_mail as _SM
import Plot_Graphs_with_Sliders as _G
import my_tools


folder = r'M:\Users\HW Infrastructure\PLC team\ARC\Temp-Eddy\Scope records for LT spice'
file_in = 'Telem003 (original).csv'
file_out = 'Telem003.csv'
delimiter = ','
time_column = 0
skiprows = 0  # if you output the CSV via Web, there will be 20 rows of nonsense
# ##  True  ###  False ## #
text_filter = (False, ['inf'])
delete_excess_df = (True, 2)
# compare_files = True
compare_files = False


if text_filter[0]:
    file = open(folder + '\\' + file_in, "r").readlines()
    for index, line in enumerate(file):
        if any(text in line for text in text_filter[1]):
            print(f'Text found in line = {index}')
            exit()
    print('Text seems good')
    del file
if compare_files:
    df = pd.read_csv(folder + '\\' + file_in, delimiter=delimiter, skiprows=skiprows)
    dff = pd.read_csv(folder + '\\' + file_out, delimiter=delimiter, names=['Time_new', 'Vlrx_new'])
    df = pd.concat([df, dff], axis=1)
    for string in ['Time', 'Vlrx']:
        df['Diff_' + string] = df[string] - df[string + '_new']
        print(f'df["Diff_{string}"].max() = {df["Diff_" + string].max()}')
        print(f'df["Diff_{string}"].min() = {df["Diff_" + string].min()}')
        print(f'df["Diff_{string}"].mean() = {df["Diff_" + string].mean()}')
else:
    df = pd.read_csv(folder + '\\' + file_in, delimiter=delimiter, dtype=object, skiprows=skiprows)
    if delete_excess_df[0] and df.shape[1] > delete_excess_df[1]:
        df = df.iloc[:, :delete_excess_df[1]]
    mana = 2760000
    sex = 4000000
    df = df[mana - mana:(mana - mana) + sex + 1].reset_index()
    del df['index']
    x = list(df.iloc[:, time_column].squeeze())
    time = np.linspace(0, float(x[-1]) - float(x[0]), len(df))
    print(f'DataFrame is ready; t-start = {0:.6f} and t-stop = {float(x[-1]) - float(x[0]):.6f}')
    del x
    time = pd.DataFrame([f'{t:.6f}' for t in time], dtype=object)
    df = pd.concat([time, df], ignore_index=True, axis=1)
    del time
    del df[1]
    # df = df.drop(df.columns[1], axis=1)
    print(f'Creating CSV of shape = {df.shape}')
    df.to_csv(folder + '\\' + file_out, header=False, index=False)
