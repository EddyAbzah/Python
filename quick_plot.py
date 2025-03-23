import my_tools
import Plot_Graphs_with_Sliders as _G
from my_pyplot import omit_plot as _O, plot as _P, print_chrome as _PC, clear as _PP, print_lines as _PL, send_mail as _SM

import io, gc, os, re, sys, glob, math, time, cmath, pylab, queue, scipy, shutil, struct, asyncio, filecmp, inspect, pathlib, smtplib
import datetime, requests, warnings, threading, functools, matplotlib, statistics, webbrowser, progressbar, multipledispatch, fnmatch, difflib
import pandas as pd, plotly as py, openpyxl as px, numpy as np, plotly.express as px, scipy.signal as signal, plotly.graph_objs as go
import matplotlib.pyplot as plt, plotly.graph_objects as go, plotly.figure_factory as ff


folder_paths = [r""]
patterns = ["*.txt", "*.csv"]
keep_columns = [True, [""]]
rename_columns = [False, {"": ""}]
remove_unnamed_column = True
add_sample_rate = [False, 50e3 / 3]
output_html__auto_open = [True, True]
output_fft__auto_open__output_csv = [False, False, False]
remove_fft_dc = True
file_paths = [os.path.join(folder_path, f) for folder_path in folder_paths for f in os.listdir(folder_path) if any(fnmatch.fnmatch(f, pattern) for pattern in patterns)]


for file_index, file_path in enumerate(file_paths):
    print(f"file_index = {file_index + 1}: {file_path}")
    # df = pd.read_csv(file_path, encoding='unicode_escape')
    df = pd.read_csv(file_path, encoding='utf-8')
    print(f"df.columns = {list(df.columns)}")
    if keep_columns[0]:
        df = df[keep_columns[1]]
    elif remove_unnamed_column:
        df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    if rename_columns[0]:
        df = df.rename(columns=rename_columns[1])
    if keep_columns[0] or rename_columns[0]:
        print(f"df.columns NEW = {list(df.columns)}")
    columns_lower = {col.lower(): col for col in df.columns}
    closest_match = difflib.get_close_matches("freq", columns_lower.keys(), n=1, cutoff=0.4)
    if closest_match:
        df = df.set_index(columns_lower[closest_match[0]])
    if add_sample_rate[0]:
        df['time'] = np.arange(len(df)) / add_sample_rate[1]
        df = df.set_index('time')

    if output_html__auto_open[0]:
        _PC(df, path=str(pathlib.Path(file_path).parent), file_name=pathlib.Path(file_path).stem, title=pathlib.Path(file_path).stem, auto_open=output_html__auto_open[1])
        print(f"Output HTML to {file_path[:-4] + ".html"}")

    if output_fft__auto_open__output_csv[0]:
        fft_data = {}
        frequencies = None
        for column in df.columns:
            signal = pd.to_numeric(df[column], errors='coerce')
            fft_values = np.fft.fft(signal)
            freqs = np.fft.fftfreq(len(fft_values), d=1 / add_sample_rate[1])
            fft_data[column] = np.abs(fft_values)[:len(freqs) // 2]
            if frequencies is None:
                frequencies = freqs[:len(freqs) // 2]

        df_fft = pd.DataFrame(fft_data, index=frequencies)
        df_fft.index.name = 'Frequency (Hz)'
        if remove_fft_dc:
            df_fft = df_fft.iloc[1:]
        if output_fft__auto_open__output_csv[2]:
            df_fft.to_csv(file_path[:-4] + " - FFT.csv")
            print(f"Output CSV to {file_path[:-4] + " - FFT.csv"}")
        _PC(df_fft, path=str(pathlib.Path(file_path).parent), file_name=pathlib.Path(file_path).stem + " - FFT", title=pathlib.Path(file_path).stem + " - FFT", auto_open=output_fft__auto_open__output_csv[1])
        print(f"Output HTML to {file_path[:-4] + " - FFT.html"}")
