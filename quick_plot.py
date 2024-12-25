import my_tools
import Plot_Graphs_with_Sliders as _G
from my_pyplot import omit_plot as _O, plot as _P, print_chrome as _PC, clear as _PP, print_lines as _PL, send_mail as _SM

import io, gc, os, re, sys, glob, math, time, cmath, pylab, queue, scipy, shutil, struct, asyncio, filecmp, inspect, pathlib, smtplib
import datetime, requests, warnings, threading, functools, matplotlib, statistics, webbrowser, progressbar, multipledispatch
import pandas as pd, plotly as py, openpyxl as px, numpy as np, plotly.express as px, scipy.signal as signal, plotly.graph_objs as go
import matplotlib.pyplot as plt, plotly.graph_objects as go, plotly.figure_factory as ff


folder_paths = [r""]
pattern = re.compile(r"")
output_html__auto_open = [True, True]
file_paths = [os.path.join(folder_path, f) for folder_path in folder_paths for f in os.listdir(folder_path) if pattern.match(f)]


for file_index, file_path in enumerate(file_paths):
    print(f"file_index = {file_index + 1}: {file_path}")
    df = pd.read_csv(file_path)
    print(f"df.columns = {list(df.columns)}")
    if output_html__auto_open[0]:
        _PC(df, path=str(pathlib.Path(file_path).parent), file_name=pathlib.Path(file_path).stem, title=pathlib.Path(file_path).stem, auto_open=output_html__auto_open[1])
        print(f"Output HTML to {str(pathlib.Path(file_path).with_suffix('.html'))}")
