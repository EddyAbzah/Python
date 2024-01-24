from io import StringIO
import pandas as pd
import plotly
import plotly.graph_objects as go
# import pyinstaller as pyinstaller
from plotly.subplots import make_subplots
import numpy as np
import log_file
from my_pyplot import omit_plot as _O, plot as _P, print_chrome as _PC, clear as _PP, print_lines as _PL, send_mail as _SM
import Plot_Graphs_with_Sliders as _G
import my_tools


# # Folders and filters:
path_output = r'V:\HW_Infrastructure\Analog_Team\ARC_Data\Results\Jupiter\Jupiter+ Improved (7E0872F4-EC)\F1 Scores with V-DC (18-08-2021)'
path_log_folder = r'C:\Users\eddy.a\Downloads\Jupiter Vdc 750v vs 850v\Data CVSs'
path_log_name = 'only holand 14-27.csv'
path_output = path_log_folder
plot_columns = 1
# # Chrome action for plotly:
plot_offline = True
kill_chrome = False and plot_offline
auto_open_chrome_output = True and plot_offline
alpha = 0.0001
df_before_alpha_index = 3
cut_from = 118000
read_plot_titles = False
plot_titles = ['V-Bank1', 'V-Bank2', 'V-DC', f'V-Bank1 (alpha = {alpha})', f'V-Bank2 (alpha = {alpha})', f'V-DC (alpha = {alpha})']
index_col = False       # for the pd.read_csv()



def main():
    global plot_titles
    list_df = []
    list_titles = []
    df = pd.read_csv(f'{path_log_folder}/{path_log_name}').dropna(how='all', axis='columns')
    plot_rows = int(len(df.columns) / plot_columns)
    if read_plot_titles:
        plot_titles = list(df.columns.values)
    all_specs = np.reshape([[{"secondary_y": False}] for x in range(len(plot_titles))], (plot_rows, plot_columns)).tolist()
    fig = make_subplots(subplot_titles=plot_titles, rows=plot_rows, cols=plot_columns, specs=all_specs, shared_xaxes=True)
    stam_index = 0
    for index_df, (title_df, sub_df) in enumerate(df.items()):
        if index_df < df_before_alpha_index:
            fig.add_trace(go.Scatter(y=sub_df[cut_from:], name=title_df, showlegend=True),
                          col=index_df % plot_columns + 1, row=int(index_df / plot_columns) + 1)
            list_df.append(sub_df)
            list_titles.append(title_df)
        else:
            sub_df_filtered = log_file.alpha_beta_filter(list_df[stam_index], alpha)[cut_from:]
            fig.add_trace(go.Scatter(y=sub_df_filtered, name=list_titles[stam_index] + f' (alpha = {alpha})',
                                     showlegend=True), col=index_df % plot_columns + 1, row=int(index_df / plot_columns) + 1)
            stam_index += 1

    print("Done")
    if plot_offline:
        plotly.offline.plot(fig, config={'scrollZoom': True, 'editable': True}, filename=f'{path_output}/{path_log_name[:-4]}.html', auto_open=auto_open_chrome_output)


if __name__ == "__main__":
    main()
