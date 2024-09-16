import math
import inspect
import numpy as np
import pandas as pd
from scipy.fft import fft
from my_pyplot import omit_plot as _O, plot as _P, print_chrome as _PC, clear as _PP, print_lines as _PL, send_mail as _SM
import Plot_Graphs_with_Sliders as _G
import my_tools

auto_open_html = False
conversion = lambda n: 10 * math.log10((n ** 2) / 0.05)

folder = r"M:\Users\Eddy A\Venus 4\Scope measurements - LF" + "\\"
files = ["3.1.c LF-RX noise floor",
         "3.1.5.d.01 RX1 NF=100mVpp", "3.1.5.d.02 RX1 NF=200mVpp", "3.1.5.d.03 RX1 NF=400mVpp",
         "3.1.5.d.04 RX2 NF=100mVpp", "3.1.5.d.05 RX2 NF=200mVpp", "3.1.5.d.06 RX2 NF=400mVpp",
         "3.1.5.d.07 RX3 NF=100mVpp", "3.1.5.d.08 RX3 NF=200mVpp", "3.1.5.d.09 RX3 NF=400mVpp"]
col_time = "TIME"
col_data = "CH2"

for file in files:
    df = pd.read_csv(folder + file + ".csv", header=0, delimiter=',', skiprows=20).dropna(how='all', axis='columns')

    freq = np.fft.fftfreq(len(df), d=df[col_time][1] - df[col_time][0])
    freq = freq[:len(freq) // 2]

    df = df.set_index(col_time)
    fft_result = fft(np.array(df[col_data]))
    magnitude = np.abs(fft_result)
    magnitude = magnitude[:len(magnitude) // 2]

    df = pd.DataFrame({"Frequency": freq, "Magnitude": magnitude})
    df = df.set_index("Frequency")
    df = df.map(conversion)

    _PC(df, path=folder, file_name=file, title=file + " (" + inspect.getsourcelines(conversion)[0][0].strip() + ")", auto_open=auto_open_html)
