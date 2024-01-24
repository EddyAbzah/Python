import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from my_pyplot import omit_plot as _O, plot as _P, print_chrome as _PC, clear as _PP, print_lines as _PL, send_mail as _SM
import Plot_Graphs_with_Sliders as _G
import my_tools

# ####   True   ###   False   #### #
plot_html__auto_open = [True, True]
spectrum = [None, 'Full', 'Zoom', 'Hi-Res'][0]
histnorm = [None, 'percent', 'probability'][2]
main_path = r'V:\HW_Infrastructure\Analog_Team\ARC_Data\Results\Venus3\6kW-7497F876 MLCC\New 370Vdc tests 01' + '\\'
# main_path = r'C:\Users\eddy.a\Downloads\VDC' + '\\'
path_in = main_path + fr'FAs - Summary\V3 Vdc370 FAs - all events (filtered).csv'
path_out = main_path + r'FAs - Summary\Histograms 07'
# path_out = main_path + r'FAs - Summary'
df = pd.read_csv(path_in).dropna(how='all', axis='columns')
plot_rows = 1

folders = ['01 KA1', '02 KA2', '04 Different solutions 01', '05 Different solutions 02', '06 5kW + OverPower'][2:4]
folders = ['01 KA1', '04 Different solutions 01', '05 Different solutions 02']
KA = ['KA 01', 'KA 02'][0]
scenarios = ['Scenario 01', 'Scenario 06', 'Scenario 08', 'Scenario 25'][0]
vdc_p302 = ['Normal (0.95)', 'New (0.99)']
P1371 = [0.5, 2, 5, 10]
P832 = [750, 1000, 1500]
P1376 = [1000, 1500, 2000]
plot_titles = ['Power Before', 'Power Diff', 'Phase', 'Abs Amps Before', 'Amp abs', 'Amp ratio']


if spectrum == 'Full':
    bins = {'Power Before': dict(start=-6666, end=6666, size=133.32),
            'Power Diff': dict(start=-6666, end=6666, size=133.32),
            'Phase': dict(start=0, end=10, size=0.1),
            'Abs Amps Before': dict(start=0, end=60000, size=600),
            'Amp abs': dict(start=0, end=4000, size=40),
            'Amp ratio': dict(start=0, end=10, size=0.1)}
elif spectrum == 'Zoom':
    bins = {'Power Before': dict(start=-600, end=600, size=12),
            'Power Diff': dict(start=-600, end=600, size=12),
            'Phase': dict(start=0, end=3, size=0.03),
            'Abs Amps Before': dict(start=0, end=60000, size=600),
            'Amp abs': dict(start=0, end=2000, size=20),
            'Amp ratio': dict(start=0, end=5, size=0.05)}
elif spectrum == 'Hi-Res':
    bins = {'Power Before': dict(start=-6666, end=6666, size=13.332),
            'Power Diff': dict(start=-6666, end=6666, size=13.332),
            'Phase': dict(start=0, end=10, size=0.01),
            'Abs Amps Before': dict(start=0, end=60000, size=60),
            'Amp abs': dict(start=0, end=4000, size=4),
            'Amp ratio': dict(start=0, end=10, size=0.01)}
else:
    bins = {'Power Before': None, 'Power Diff': None, 'Phase': None, 'Abs Amps Before': None, 'Amp abs': None, 'Amp ratio': None}


df = df.loc[(df['Folder'].isin(folders)) & (df['KA'] == KA) & (df['Scenario'] == scenarios)]
for plot_title in plot_titles:
    plot_main_title = f'{plot_title} - Complete Histogram (bins={spectrum} & histnorm={histnorm})'
    plot_counter = 0
    if plot_rows == 1:
        fig = make_subplots(rows=1, cols=1)
    else:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=['Normal Vdc (P302 = 0.95)', 'New Vdc (P302 = 0.99)'])

    name = 'Normal Vdc (P302 = 0.95)'
    sdf = df.loc[(df['Vdc (P302)'] == 'Normal (0.95)')][plot_title]
    fig.add_trace(go.Histogram(x=sdf, name=name, xbins=bins[plot_title], histnorm=histnorm), row=1, col=1)
    steps = [dict(method="update", label="all", args=[{"visible": [True] * 13}, {"title": f'{plot_title}: {plot_main_title.split(" - ")[0]} - All'}], )]

    for p1371 in P1371:
        for p832 in P832:
            for p1376 in P1376:
                name = f'P1371={p1371}; P832={p832}; P1376={p1376}'
                sdf = df.loc[(df['Vdc (P302)'] == 'New (0.99)') & (df['P1371'] == p1371) & (df['P832'] == p832) & (df['P1376'] == p1376)][plot_title]
                if sdf.empty:
                    if plot_html__auto_open[0]:
                        print(f'name = \"{name}\" is not found')
                    continue
                if plot_rows == 1:
                    fig.add_trace(go.Histogram(x=sdf, name=name, xbins=bins[plot_title], histnorm=histnorm), row=1, col=1)
                else:
                    fig.add_trace(go.Histogram(x=sdf, name=name, xbins=bins[plot_title], histnorm=histnorm), row=2, col=1)

                plot_counter += 1
                visible = [True] + [False] * 12
                visible[plot_counter] = True
                step = dict(method="update", label=name, args=[{"visible": visible},  {"title": f'{plot_title}: {plot_main_title.split(" - ")[0]} - {name}'}], )
                steps.append(step)

    sliders = [dict(active=10, currentvalue={"prefix": "Plot: "}, pad={"t": 50}, steps=steps)]
    fig.update_layout(title=f'{plot_main_title}', sliders=sliders)       # autosize=True, height=4000,
    fig.update_xaxes(rangeslider=dict(visible=False))
    if histnorm == 'percent':
        fig.update_yaxes(tickmode='array', tickvals=list(range(0, 100, 1)))
    if plot_html__auto_open[0]:
        fig.write_html(f'{path_out}\\{plot_main_title}.html', auto_open=plot_html__auto_open[1])
