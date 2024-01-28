import os
import re
import pandas as pd
from math import log
from io import StringIO
import Library_Functions
if os.getlogin() == "eddy.a":
    from my_pyplot import omit_plot as _O, plot as _P, print_chrome as _PC, clear as _PP, print_lines as _PL, send_mail as _SM
    import Plot_Graphs_with_Sliders as _G
    import my_tools


regex_pattern = '[^ ()a-zA-Z0-9,:%#_+°∞πΣµΩΩ±≥≤≈√²—-]'
cut_all_plots = [True, 0, 10e6]


def get_df(path, traces=['Magnitude'], trace_sep='\n\n', apply_log=False):
    """
    Get a CSV exported from Bode, and return a pandas DataFrame.
    Args:
        path: String - Full file path with extension.
        traces: List of strings - traces (data) to be returned; default = ['Magnitude'].
        trace_sep: String - how different measurements are separated in the file; default = '\n\n'.
        apply_log: Bool - apply a simple log(x) to the data; default = False.
    Returns:
        df_full (pandas DataFrame).
    """
    print(f'Bode_to_DF.get_df: {path = }, {traces = }, {trace_sep = }, {apply_log = }')
    df_full = pd.DataFrame()
    with open(path, 'r') as file:
        for i, sub_file in enumerate(file.read().split(trace_sep)):
            full_title = re.sub(regex_pattern, '', sub_file.split('\n')[0]).replace('Magnitude ()', 'Magnitude (Ω)')
            if full_title is not None and full_title != '':
                full_title = ','.join([full_title.split(',')[0].split(': ')[1]] + [b[0] + ' - ' + b[-1] for b in [a.split(': ') for a in full_title.split(',')[1:]]])
                print(f'trace number {i:00}: titles = {full_title}')
                df = pd.read_csv(StringIO(full_title + sub_file[sub_file.find('\n'):]), index_col=0).dropna(how='all', axis='columns')
                if traces is not None and len(traces) > 0:
                    df = df[[col for col in df.columns if any([ts in col for ts in traces])]]
                if apply_log:
                    df.iloc[:, 0] = df.iloc[:, 0].apply(lambda x: log(x))
                df_full = pd.concat([df_full, df], axis=1)
    return df_full


if __name__ == "__main__":
    folder = r"M:\Users\HW Infrastructure\PLC team\ARC\Temp-Eddy\Jupiter48\LTspice vs Bode\\"
    file_path = folder + "Bode Impedance.csv"
    df1 = get_df(file_path, apply_log=True)
    df1.to_csv(file_path[:-4] + ' (Pivot)' + file_path[-4:])

    file_path = folder + "Bode Transmission.csv"
    df2 = get_df(file_path)
    df2.to_csv(file_path[:-4] + ' (Pivot)' + file_path[-4:])

    df1.rename(lambda title: 'Impedance ' + title, axis='columns', inplace=True)
    df2.rename(lambda title: 'Transmission ' + title, axis='columns', inplace=True)
    df_all = pd.concat([df1, df2])
    if cut_all_plots[0]:
        df_all = df_all[df_all.index > cut_all_plots[1]]
        df_all = df_all[df_all.index < cut_all_plots[2]]
    df_all.to_csv(folder + 'Bode - all measurements.csv')
    df_all.sort_index().to_csv(folder + 'Bode - all measurements (sorted).csv')
    Library_Functions.print_chrome(df_all, folder, 'Bode - all measurements')
