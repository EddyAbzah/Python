import os
import sys
import glob
import math
import plotly
import inspect
import pathlib
import smtplib
import statistics
import numpy as np
import pandas as pd
from enum import Enum
from scipy import signal
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from my_pyplot import omit_plot as _O, plot as _P, print_chrome as _PC, clear as _PP, print_lines as _PL, send_mail as _SM
import Plot_Graphs_with_Sliders as _G
import my_tools

# ####   True   ###   False   #### #
output_text = False
if output_text:
    default_stdout = sys.stdout
    sys.stdout = open(r'C:\Users\eddy.a\Downloads\VDC\Terminal Log 01.txt', 'w')

# df = pd.DataFrame()
# mana = ['default S1+2+3', 'default S1+2', 'default S1', 'high S1', 'high S1+2', 'high S1+2+3', 'default S2', 'default S3']
path_in = r"V:\HW_Infrastructure\Analog_Team\ARC_Data\Results\Jupiter\Jupiter+ Improved 2024 (7B0F73A7-A4)\Arc Full FAs 04 - AC side (02-01-2024)"
files = glob.glob(f'{path_in}\\*spi*.txt')
files.sort()
for file in files:
    df = pd.read_csv(file).dropna(how='all', axis='columns')
    # df = my_tools.convert_df_counters(df)
    print(f'df.columns = {", ".join(list(df.columns))}')
    df.index = df.index / 16667
    path_out = path_in + r'\RAW Plots'
    for t, sub_p in zip(["RXout + Vdc", "Vac + Iac"], [["RXout", "Vdc"], ["Vac1", "Vac2", "Iac1"]]):
        dff = df[sub_p]
        name = "Rec" + file.split("Rec")[-1][:7] + " " + t + " - " + file.split("Str ")[-1][:-4]
        _PC(dff, path=path_out, file_name=name, title=name, auto_open=False)


if output_text:
    sys.stdout.close()
    sys.stdout = default_stdout
