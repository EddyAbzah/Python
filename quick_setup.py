import my_tools
import Plot_Graphs_with_Sliders as _G
from my_pyplot import omit_plot as _O, plot as _P, print_chrome as _PC, clear as _PP, print_lines as _PL, send_mail as _SM

import io, gc, os, re, sys, glob, math, time, cmath, pylab, queue, scipy, shutil, struct, asyncio, filecmp, inspect, pathlib, smtplib
import datetime, requests, warnings, threading, functools, matplotlib, statistics, webbrowser, progressbar, multipledispatch
import pandas as pd, plotly as py, openpyxl as px, numpy as np, plotly.express as px, scipy.signal as signal, plotly.graph_objs as go
import matplotlib.pyplot as plt, plotly.graph_objects as go, plotly.figure_factory as ff

file_path1 = r""
file_path2 = r""
if file_path1 != "":
    df1 = pd.read_csv(file_path1)
    print(f'df1.columns = {list(df1.columns)}')
if file_path2 != "":
    df2 = pd.read_csv(file_path2)
    print(f'df2.columns = {list(df2.columns)}')
exit()
