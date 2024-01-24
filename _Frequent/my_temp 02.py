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

path_in = r"V:\HW_Infrastructure\Analog_Team\ARC_Data\Results\Jupiter\Jupiter+ Improved 2024 (7B0F73A7-A4)\Arc Full FAs 04 - AC side (02-01-2024)\Full FA Test 04 Rec095 spi 1[A] x1Str 10% Sag.txt"
df = pd.read_csv(path_in).dropna(how='all', axis='columns')
# df = my_tools.convert_df_counters(df)
print(f'df.columns = {", ".join(list(df.columns))}')
df.index = df.index / 16667
print()
# 'RXout','Vac1','Vac2','Iac1','IacRMS','Vdc'
