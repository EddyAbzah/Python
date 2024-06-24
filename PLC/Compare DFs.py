import os
import pandas as pd
import Library_Functions
if os.getlogin() == "eddy.a":
    from my_pyplot import omit_plot as _O, plot as _P, print_chrome as _PC, clear as _PP, print_lines as _PL, send_mail as _SM
    import Plot_Graphs_with_Sliders as _G
    import my_tools

folder = r"C:\Users\eddy.a\Downloads"
file_out = ""
file_path_1 = folder + "\\" + ".csv"
file_path_2 = folder + "\\" + ".csv"
set_index = [True, "Freq [kHz]"]
interpolate = [['linear', None], ['polynomial', 2]][0]

# Read DFs and set indexes
df1 = pd.read_csv(file_path_1)
df2 = pd.read_csv(file_path_2)
if set_index[0]:
    df1 = df1.set_index(set_index[1])
    df2 = df2.set_index(set_index[1])

# Combine indices from both DataFrames and reindex both DataFrames
combined_index = df1.index.union(df2.index)
df1_reindexed = df1.reindex(combined_index)
df2_reindexed = df2.reindex(combined_index)

# Interpolate missing values using linear interpolation
df1_interpolated = df1_reindexed.interpolate(method=interpolate[0], order=interpolate[1])
df2_interpolated = df2_reindexed.interpolate(method=interpolate[0], order=interpolate[1])

# Plot
Library_Functions.print_chrome(df1_interpolated - df2_interpolated, folder, file_out, scale='lin')
