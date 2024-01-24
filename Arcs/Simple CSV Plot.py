import plotly
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from my_pyplot import plot as _P, print_chrome as _PC, clear as _PP, print_lines as _PL, send_mail as _SM
import Plot_Graphs_with_Sliders as _G
import my_tools


# ###   True   ###   False   ### #
# # Chrome action for plotly:
plot_offline = True
kill_chrome = False and plot_offline
auto_open_chrome_output = True and plot_offline
# # Files:
path_output = r'M:\Users\HW Infrastructure\PLC team\ARC\Temp-Eddy'
path_file_folder = r'M:\Users\HW Infrastructure\PLC team\ARC\Temp-Eddy'
path_file_name = 'Spectrum 1kHz-260kHz'
path_output_file_and_title = 'Spectrum 1kHz-260kHz'
# # Output:
first_column_is_x_axis = [True, np.linspace(1e3, 260e3, num=1001)]
first_step_is_plot_all = (True, 'All frequencies')
range_x_axis = (False, 'log', False, [0, 475])
range_y_axis = (False, 'log', False, [-19, 17])
min_max_avg = (False, False, False)


df = pd.read_csv(f'{path_file_folder}//{path_file_name}.csv')
fig = make_subplots(rows=1, cols=1, shared_xaxes=True)
slider_steps = []
step_len = 1
for value in min_max_avg:
    if value:
        step_len += 1
step_len = len(df) * step_len
if first_step_is_plot_all[0]:
    step = dict(args=[{"visible": [True] * step_len}, ], label=first_step_is_plot_all[1])
    slider_steps.append(step)

index_plot = 0
if first_column_is_x_axis[0]:
    x_axis = df.pop(list(df.head(0))[0])
else:
    if len(first_column_is_x_axis) == 1 or first_column_is_x_axis[1] is None:
        x_axis = None
    else:
        x_axis = first_column_is_x_axis[1]
for plot_name, plot in df.iteritems():
    visible = first_step_is_plot_all[0] or index_plot == 0
    fig.add_trace(go.Scatter(x=x_axis, y=plot, name=plot_name, visible=first_step_is_plot_all[0], showlegend=True), col=1, row=1)
    if min_max_avg[0]:
        fig.add_trace(go.Scatter(x=[x_axis.iloc[plot.idxmin()]], y=[plot.min()], name=f'{plot_name} - Min', visible=visible, showlegend=True), col=1, row=1)
    if min_max_avg[1]:
        fig.add_trace(go.Scatter(x=[x_axis.iloc[plot.idxmax()]], y=[plot.max()], name=f'{plot_name} - Max', visible=visible, showlegend=True), col=1, row=1)
    if min_max_avg[2]:
        fig.add_trace(go.Scatter(x=x_axis, y=[plot.mean()]*len(plot), name=f'{plot_name} - Avg', visible=visible, showlegend=True), col=1, row=1)
    step = dict(args=[{"visible": [False] * step_len}, ], label=plot_name)
    step["args"][0]["visible"][index_plot] = True
    for value in [True, *min_max_avg]:
        if value:
            step["args"][0]["visible"][index_plot] = True
            index_plot += 1
    slider_steps.append(step)

if plot_offline:
    if range_x_axis[0]:
        fig.update_xaxes(type=range_x_axis[1])
    if range_x_axis[2]:
        fig.update_xaxes(range=range_x_axis[3])
    if range_y_axis[0]:
        fig.update_yaxes(type=range_y_axis[1])
    if range_y_axis[2]:
        fig.update_yaxes(range=range_y_axis[3])
    fig.update_layout(sliders=[dict(active=0, pad={"t": 50}, steps=slider_steps, bgcolor="#ffb200", currentvalue=dict(xanchor="center", font=dict(size=16)))])
    fig.update_layout(title=path_output_file_and_title, title_font_color="#407294", title_font_size=40, legend_title="Plots:", legend_title_font_color="green")
    plotly.offline.plot(fig, config={'scrollZoom': True, 'editable': True}, filename=f'{path_output}/{path_output_file_and_title}.html', auto_open=auto_open_chrome_output)