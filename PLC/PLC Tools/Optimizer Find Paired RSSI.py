import os
import sys
import statistics
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
if os.getlogin() == "eddy.a":
    from my_pyplot import omit_plot as _O, plot as _P, print_chrome as _PC, clear as _PP, print_lines as _PL, send_mail as _SM
    import Plot_Graphs_with_Sliders as _G
    import my_tools


# ## ### True ### ## # ## ### False ### ## #
output_text = True
output_text_path = r"M:\Users\ShacharB\Projects\PLC Leakage - RSSI Ratio Issue\RSSI Ratio Issue Gen4 - 12.2022\Solution Procedure\Analysis from 30-08-2023 to 15-10-2024\Python log.txt"
file_in = r"C:\Users\eddy.a\Downloads\Solution Procedure\RAW data.csv"
head_date = "time"
head_opt = "optimizerid"
head_rssi = "param_129"
remove_glitches = [False, 1.5, 0.6667]


if output_text:
    default_stdout = sys.stdout
    sys.stdout = open(output_text_path, 'w')

df = pd.read_csv(file_in).dropna(how='all', axis='columns')
for optimizer_index, optimizer_id in enumerate(df[head_opt].unique()):
    print(f'Searching for Pairing in Optimizer number {optimizer_index + 1}: {optimizer_id = }')
    sdf = df[df[head_opt] == optimizer_id].sort_values(head_date)
    sdf = sdf.reset_index(drop=True)
    rssi = sdf[head_rssi]
    print(f'Searching for Pairing in Optimizer number {optimizer_index + 1}: {optimizer_id = }')




if output_text:
    sys.stdout.close()
    sys.stdout = default_stdout
