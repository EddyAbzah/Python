import os
import sys
import math
import plotly
import inspect
import pathlib
import smtplib
import statistics
import numpy as np
import pandas as pd
import seaborn as sns
from enum import Enum
from scipy import signal
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from my_pyplot import plot as _P, print_chrome as _PC, clear as _PP, print_lines as _PL, send_mail as _SM
import Plot_Graphs_with_Sliders as _G
import my_tools

# ####   True   ###   False   #### #
path_in = r'M:\Users\HW Infrastructure\PLC team\ARC\Temp-Eddy\KA1.csv'
df = pd.read_csv(path_in).dropna(how='all', axis='columns')
