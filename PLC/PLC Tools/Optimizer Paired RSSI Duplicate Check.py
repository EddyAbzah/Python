import os
import math
import numpy as np
import pandas as pd
if os.getlogin() == "eddy.a":
    from my_pyplot import omit_plot as _O, plot as _P, print_chrome as _PC, clear as _PP, print_lines as _PL, send_mail as _SM
    import Plot_Graphs_with_Sliders as _G
    import my_tools

df = pd.read_csv(r"M:\Users\ShacharB\Projects\PLC Leakage - RSSI Ratio Issue\RSSI Ratio Issue Gen4 - 12.2022\Solution Procedure\236 Day Analysis\Attachments\P129 Duplicates.txt").dropna(how='all', axis='columns')


folder = r"M:\Users\ShacharB\Projects\PLC Leakage - RSSI Ratio Issue\RSSI Ratio Issue Gen4 - 12.2022\Solution Procedure\236 Day Analysis\\"
dfs = list()
dfs.append(pd.read_csv(folder + "RSSI_Issue_-_P129_no_duplicates_2023_10_09.csv").dropna(how='all', axis='columns'))
dfs.append(pd.read_csv(folder + "RSSI_Issue_-_P129_no_duplicates_2024_02_22.csv").dropna(how='all', axis='columns'))
method = 2

for idf, df in enumerate(dfs):
    print(f"\ndf number {idf + 1}:")
    match method:
        case 1:
            last_rssi = list(df["param_129"])
            duplicates = [False]
            for i in range(1, len(last_rssi)):
                duplicates.append(last_rssi[i] == last_rssi[i - 1])
            for i, v in enumerate(duplicates):
                if v:
                    print("\n" + df.iloc[i - 2:i + 3, ].to_string())

        case 2:
            pd.options.mode.chained_assignment = None
            print(f"{df["time"].nunique() = }")
            for i_time, time in enumerate(df["time"].unique()):
                sdf = df[df["time"] == time]
                if not sdf["param_129"].is_unique:
                    print(f"\n{i_time = }\n" + sdf[sdf.duplicated(subset=["param_129"], keep=False)].to_string())

        case _:
            print("bye")
