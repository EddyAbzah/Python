import os
import pandas as pd
import Bode_to_DF
import LTspice_to_DF
import Library_Functions
if os.getlogin() == "eddy.a":
    from my_pyplot import omit_plot as _O, plot as _P, print_chrome as _PC, clear as _PP, print_lines as _PL, send_mail as _SM
    import Plot_Graphs_with_Sliders as _G
    import my_tools

# Bode_to_DF:
cut_all_plots = [True, 0, 10e6]
df = pd.DataFrame()
file_path = r"C:\Users\eddy.a\Downloads\Jupiter48\LTspice vs Bode\Bode Impedance.csv"
df = pd.concat([df, Bode_to_DF.get_df(file_path, apply_log=True).rename(lambda title: 'Impedance ' + title, axis='columns')])
file_path = r"C:\Users\eddy.a\Downloads\Jupiter48\LTspice vs Bode\Bode Transmission.csv"
df = pd.concat([df, Bode_to_DF.get_df(file_path).rename(lambda title: 'Transmission ' + title, axis='columns')])

# LTspice_to_DF:
file_path = r"C:\Users\eddy.a\Downloads\Jupiter48\LTspice vs Bode\LTspice Diff.txt"
df = pd.concat([df, LTspice_to_DF.get_df(file_path).rename(lambda title: 'Diff ' + title, axis='columns')])
file_path = r"C:\Users\eddy.a\Downloads\Jupiter48\LTspice vs Bode\LTspice DC+.txt"
df = pd.concat([df, LTspice_to_DF.get_df(file_path).rename(lambda title: 'DC+ ' + title, axis='columns')])
file_path = r"C:\Users\eddy.a\Downloads\Jupiter48\LTspice vs Bode\LTspice DC-.txt"
df = pd.concat([df, LTspice_to_DF.get_df(file_path).rename(lambda title: 'DC- ' + title, axis='columns')])

# All:
if cut_all_plots[0]:
    df = df[df.index > cut_all_plots[1]]
    df = df[df.index < cut_all_plots[2]]
Library_Functions.print_chrome(df, r"C:\Users\eddy.a\Downloads\Jupiter48\LTspice vs Bode", 'Bode vs LTspice - all measurements')
