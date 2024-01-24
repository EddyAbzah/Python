import plotly
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots

from my_pyplot import omit_plot as _O, plot as _P, print_chrome as _PC, clear as _PP, print_lines as _PL, send_mail as _SM
import Plot_Graphs_with_Sliders as _G
import my_tools

### ## #   True or False   # ## ###
folder = r'V:\HW_Infrastructure\Analog_Team\ARC_Data\Results\Venus3\6kW-7497F876 MLCC\New 370Vdc tests 02\Arcs for Vdc 07 (07-11-2023)'
file_in = 'Arcs for Vdc 07.xlsx'
excel_sheet__range = ['Test_Nonstandard_Arcs', 1000]
cut_max_power__range = [False, -65, 10]

df = pd.read_excel(folder + '\\' + file_in, sheet_name=excel_sheet__range[0], usecols=range(5, excel_sheet__range[1]))
df = df.set_index(df.iloc[:, 0]).T
cols = [s for s in list(df.head(0)) if isinstance(s, str) and 'Zes' in s and 'Time' not in s]
df = df[cols].dropna().reset_index().drop(0).reset_index()
df = df.drop(list(df.head(0))[:2], axis=1)

fig = make_subplots(rows=3, cols=1, shared_xaxes=True, subplot_titles=['Zes Power [W]', 'Zes Current [A]', 'Zes Voltage [V]'])
visible = [True] * df.shape[1]
slider_steps = [dict(method="update", args=[{"visible": visible}, {"title": "All records"}], label='All records')]
for i in range(0, df.shape[1], 3):
    visible = [False] * df.shape[1]
    for j in range(3):
        sdf = pd.to_numeric(df.iloc[:, i + j])
        if cut_max_power__range[0]:
            max_index = sdf.idxmax()
            sdf = sdf[max_index + cut_max_power__range[1]:max_index + cut_max_power__range[2]]
        fig.add_trace(go.Scatter(y=sdf, name=list(df.head(0))[i + j][3:], visible=True, showlegend=True), col=1, row=j + 1)
        visible[i + j] = True
    step = dict(method="update", args=[{"visible": visible}, {"title": 'Rec' + list(df.head(0))[i][-3:]}], label='Rec' + list(df.head(0))[i][-3:])
    slider_steps.append(step)
fig.update_layout(sliders=[dict(active=0, pad={"t": 50}, steps=slider_steps, bgcolor="#ffb200", currentvalue=dict(xanchor="center", font=dict(size=16)))])
fig.update_layout(title=file_in[:-5], title_font_color="#407294", title_font_size=40, legend_title="Plots:", legend_title_font_color="green")
plot_name = f'{folder}\\{file_in[:-5]} - Zes {"cut" if cut_max_power__range[0] else "Full"}.html'
plotly.offline.plot(fig, config={'scrollZoom': True, 'editable': True}, filename=plot_name, auto_open=True)
