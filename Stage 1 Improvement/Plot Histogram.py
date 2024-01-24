import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from my_pyplot import omit_plot as _O, plot as _P, print_chrome as _PC, clear as _PP, print_lines as _PL, send_mail as _SM
import Plot_Graphs_with_Sliders as _G
import my_tools

# ####   True   ###   False   #### #
spectrum = ['Full', 'Zoom', None][0]
histnorm = ['percent', 'probability', None][0]
path_in = r'M:\Users\HW Infrastructure\PLC team\ARC\Temp-Eddy\V3 FAs - Summary\V3 Vdc370 FAs - all records.csv'
# df = pd.read_excel(path_in, sheet_name='Table').dropna(how='all', axis='columns')
df = pd.read_csv(path_in).dropna(how='all', axis='columns')
plots = ['Folder', 'KA', 'Rec', 'Scenario', 'Vdc (P302)', 'P1371', 'P832', 'P1376', 'Power Diff', 'Bitmap', 'Phase', 'Amp abs', 'Amp ratio']

KA = ['KA1', 'KA2']
scenarios = ['Scenario 01', 'Scenario 06', 'Scenario 08', 'Scenario 25']
vdc_p302 = ['Normal (0.95)', 'New (0.99)']
P1371 = [0.5, 2, 5, 10]
P832 = [750, 1000, 1500]
P1376 = [1000, 1500, 2000]
plot_titles = ['Power Diff', 'Phase', 'Amp abs', 'Amp ratio']

plot_split = [KA, scenarios, vdc_p302, P1371, P832, P1376][0]
plot_main_title = f'Complete Histogram - {spectrum}'
path_out = fr'M:\Users\HW Infrastructure\PLC team\ARC\Temp-Eddy\V3 FAs - Summary\{plot_main_title}.html'


fig = make_subplots(rows=len(plot_split), cols=1, shared_xaxes=True, subplot_titles=[f'plot_split = {n}' for n in plot_split])
if spectrum == 'Full':
    bins = {'Power Diff': dict(start=-6666, end=6666, size=133.32), 'Phase': dict(start=0, end=10, size=0.1), 'Amp abs': dict(start=0, end=4000, size=40), 'Amp ratio': dict(start=0, end=10, size=0.1)}
if spectrum == 'Zoom':
    bins = {'Power Diff': dict(start=-600, end=600, size=12), 'Phase': dict(start=0, end=3, size=0.03), 'Amp abs': dict(start=0, end=2000, size=20), 'Amp ratio': dict(start=0, end=5, size=0.05)}
else:
    bins = None
for i0, v0 in enumerate(KA):
    for i1, v1 in enumerate(plot_titles):
        for i2, v2 in enumerate(scenarios):
            for i3, v3 in enumerate(vdc_p302):
                # name = f'{v0}; S{v2.split(" ")[-1]}; v{v3.split("(")[-1][:-1]}'
                # if spectrum == 'Vdc':
                #     fig.add_trace(go.Histogram(x=df.loc[(df['KA'] == v0) & (df['Scenario'] == v2) & (df['Vdc (P302)'] == v3)][v1], name=name, xbins=bins[v1], histnorm=histnorm), row=i3+1, col=1)
                # elif spectrum == 'Scenarios':
                #     fig.add_trace(go.Histogram(x=df.loc[(df['KA'] == v0) & (df['Scenario'] == v2) & (df['Vdc (P302)'] == v3)][v1], name=name, xbins=bins[v1], histnorm=histnorm), row=i2+1, col=1)
                # else:
                #     fig.add_trace(go.Histogram(x=df.loc[(df['KA'] == v0) & (df['Scenario'] == v2) & (df['Vdc (P302)'] == v3)][v1], name=name, xbins=bins[v1], histnorm=histnorm), row=1, col=1)
                for i4, v4 in enumerate(P832):
                    for i5, v5 in enumerate(P1376):
                        name = f'{v0}; P832={v4}; P1376={v5}'
                        fig.add_trace(go.Histogram(x=df.loc[(df['KA'] == v0) & (df['Scenario'] == v2) & (df['Vdc (P302)'] == v3) & (df['P832'] == v4) & (df['P1376'] == v5)][v1]
                                                   , name=name, xbins=bins[v1], histnorm=histnorm), row=i4 + 1, col=1)

steps = [dict(method="update", label="all", args=[{"visible": [True] * len(fig.data)},  {"title": plot_main_title}], )]
step_index = 0
for i0, v0 in enumerate(KA):
    for i1, v1 in enumerate(plot_titles):
        visible = [False] * len(fig.data)
        # for i2 in range(len(scenarios)):
        #     for i3 in range(len(vdc_p302)):
        #         visible[step_index] = True
        #         step_index += 1
        for i4, v4 in enumerate(P832):
            for i5, v5 in enumerate(P1376):
                visible[step_index] = True
                step_index += 1
        step = dict(method="update", label=f'{v0}: {v1}', args=[{"visible": visible},  {"title": plot_main_title + ': ' + v1}], )
        steps.append(step)
sliders = [dict(active=10, currentvalue={"prefix": "Plot: "}, pad={"t": 50}, steps=steps)]

fig.update_layout(title=plot_main_title, sliders=sliders)       # autosize=True, height=4000,
fig.update_xaxes(rangeslider=dict(visible=False))
fig.write_html(path_out, auto_open=True)
