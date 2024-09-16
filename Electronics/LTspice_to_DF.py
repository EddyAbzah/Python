import os
import math
import pandas as pd
from math import log
from io import StringIO
import Library_Functions
if os.getlogin() == "eddy.a":
    from my_pyplot import omit_plot as _O, plot as _P, print_chrome as _PC, clear as _PP, print_lines as _PL, send_mail as _SM
    import Plot_Graphs_with_Sliders as _G
    import my_tools

conversion = {'volt_rms': lambda n1, n2: (10 ** (n1 / 20)) / (2 * math.sqrt(2)),
              'dbm': lambda n1, n2: 10 * math.log10((n1 ** 2) / 0.05),
              'log': lambda n1, n2: log(n1),
              'add': lambda n1, n2: n1 + n2,
              'mul': lambda n1, n2: n1 * n2}
cut_all_plots = [True, 0, 10e6]


def get_df(path, traces=['Magnitude'], apply_map='off', map_coefficient=None):
    """
    Get a CSV exported from LTspice, and return a pandas DataFrame.
    Real and imaginary parts will be separated and prefixed with "Magnitude" and "Phase".
    Args:
        path: String - Full file path with extension.
        traces: List of strings - traces (data) to be returned; default = ['Magnitude'].
        apply_map: String - choose to convert the values to "volt_rms", "dbm", "log", "add", or "mul"; default = 'off'.
        map_coefficient: number - choose the number for the "add" (addition) or "mul" (multiply) functions; default = None.
    Returns:
        df_full (pandas DataFrame).
    """
    print(f'LTspice_to_DF.get_df: {path = }, {traces = }, {apply_map = }, {map_coefficient = }')
    with open(path, 'r') as file:
        file = file.read()
        full_title = file.split('\n')[0].replace('Freq.', 'Frequency (Hz)').replace(',', ':').split('\t')
        full_title = full_title[0] + ',' + ','.join(['Magnitude ' + a + ',Phase ' + b for a, b in zip(full_title[1:], full_title[1:])])
        file = file.replace('\t', ',')
        file = file[file.find('\n'):].replace('(', '').replace(')', '').replace('dB', '').replace('°', '')
        df = pd.read_csv(StringIO(full_title + file), index_col=0).dropna(how='all', axis='columns')
        if traces is not None and len(traces) > 0:
            df = df[[col for col in df.columns if any([ts in col for ts in traces])]]
        if apply_map is not None and apply_map in conversion:
            df = df.apply(lambda n: conversion[apply_map](n, map_coefficient))
    return df


if __name__ == "__main__":
    folder = r"M:\Users\HW Infrastructure\PLC team\ARC\Temp-Eddy\Jupiter48\LTspice vs Bode"
    file_path = folder + "\\" + "LTspice Diff.txt"
    df1 = get_df(file_path)
    df1.to_csv(file_path[:-4] + ' (Pivot)' + file_path[-4:])

    file_path = folder + "\\" + "LTspice DC+.txt"
    df2 = get_df(file_path)
    df2.to_csv(file_path[:-4] + ' (Pivot)' + file_path[-4:])

    file_path = folder + "\\" + "LTspice DC-.txt"
    df3 = get_df(file_path)
    df3.to_csv(file_path[:-4] + ' (Pivot)' + file_path[-4:])

    file_path = file_path[:file_path.rfind('\\')]
    df1.rename(lambda title: 'Diff ' + title, axis='columns', inplace=True)
    df2.rename(lambda title: 'DC+ ' + title, axis='columns', inplace=True)
    df3.rename(lambda title: 'DC- ' + title, axis='columns', inplace=True)
    df_all = pd.concat([df1, df2, df3])
    if cut_all_plots[0]:
        df_all = df_all[df_all.index > cut_all_plots[1]]
        df_all = df_all[df_all.index < cut_all_plots[2]]
    df_all.to_csv(file_path + '\\LTspice - all measurements.csv')
    df_all.sort_index().to_csv(file_path + "\\" + "LTspice - all measurements (sorted).csv")
    Library_Functions.print_chrome(df_all, file_path, 'LTspice - all measurements')
