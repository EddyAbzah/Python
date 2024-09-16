import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from my_pyplot import plot as _P, print_chrome as _PC, clear as _PP, print_lines as _PL, send_mail as _SM
import Plot_Graphs_with_Sliders as _G
import my_tools

# ####   True   ###   False   #### #
print_errors = True
auto_open_html = True
bins = ['Automatic', 'Zoom', 'Hi-Res'][0]
histnorm = [None, 'percent', 'probability'][0]
main_path = r'M:\Users\MohamadH\Tasks\Non-Standard JPI CM_SW\MPPT + MX FAs 01 (03-10-2023)' + '\\'
main_path = r'C:\Users\eddy.a\Downloads\VDC' + '\\'
path_in = 'JPI CM FAs 01 - All events.csv'
path_in = 'V3 Vdc370 FAs - all events (filtered).csv'
path_out = 'JPI CM FAs 01 - Histogram'
path_out = 'FAs - Summary'


plot_titles = ['Power Diff', 'Bitmap', 'Phase', 'Amp abs', 'Amp ratio']
plot_titles = ['Bitmap']
rec_title = 'Rec'
# add_special_case = [False, plot_titles[4], ' zoomed-in', dict(start=5, end=5000, size=50)]
add_special_case = [False, None, ' zoomed-in', dict(start=5, end=5000, size=50)]

# if bins == 'Zoom':
#     bins = dict(zip(plot_titles, [dict(start=-600, end=600, size=12), dict(start=0, end=3, size=0.03),
#                                   dict(start=0, end=2000, size=20), dict(start=0, end=5, size=0.05)]))
# elif bins == 'Hi-Res':
#     bins = dict(zip(plot_titles, [dict(start=-6666, end=6666, size=13.332), dict(start=0, end=10, size=0.01),
#                                   dict(start=0, end=4000, size=4), dict(start=0, end=10, size=0.01)]))
# else:
#     bins = dict(zip(plot_titles, [None] * len(plot_titles)))


df = pd.read_csv(main_path + path_in).dropna(how='all', axis='columns')
fig = make_subplots(rows=1, cols=1)
plot_names = []
plot_counter = []
for plot_title in plot_titles:
    plot_counter.append(0)
    drop_list = list(frozenset(plot_titles) - {plot_title}) + [rec_title]
    sdf = df.drop(drop_list, axis=1)
    groups = sdf.groupby(list(sdf.drop(plot_title, axis=1).columns))
    for full_name, group in groups:
        name = ', '.join([f'{a} = {b}' for a, b in zip(list(frozenset(groups.head(0)) - {plot_title}), full_name)]) + f' - {plot_title}'
        if group.empty:
            if print_errors:
                print(f'name = \"{name}\" is not found')
            continue
        # fig.add_trace(go.Histogram(x=group[plot_title], name=name, xbins=bins[plot_title], histnorm=histnorm), row=1, col=1)
        fig.add_trace(go.Histogram(x=group[plot_title], name=name, xbins=None, histnorm=histnorm), row=1, col=1)
        plot_counter[-1] += 1

if add_special_case[0]:
    plot_title = add_special_case[1]
    plot_counter.append(0)
    drop_list = list(frozenset(plot_titles) - {plot_title}) + [rec_title]
    sdf = df.drop(drop_list, axis=1)
    groups = sdf.groupby(list(sdf.drop(plot_title, axis=1).columns))
    for full_name, group in groups:
        name = ', '.join([f'{a} = {b}' for a, b in zip(list(frozenset(groups.head(0)) - {plot_title}), full_name)]) + f' - {plot_title}'
        if group.empty:
            if print_errors:
                print(f'name = \"{name}\" is not found')
            continue
        # fig.add_trace(go.Histogram(x=group[plot_title], name=name, xbins=add_special_case[3], histnorm=histnorm), row=1, col=1)
        fig.add_trace(go.Histogram(x=group[plot_title], name=name, xbins=None, histnorm=histnorm), row=1, col=1)
        plot_counter[-1] += 1
    plot_titles = plot_titles + [plot_title + add_special_case[2]]

steps = [dict(method="update", label="all", args=[{"visible": [True] * sum(plot_counter)}, {"title": path_out + ': All plots'}], )]
visible_plots = 0
for i_p, p in enumerate(plot_counter):
    visible = [False] * sum(plot_counter)
    for i in range(p):
        visible[visible_plots + i] = True
    visible_plots += p
    step = dict(method="update", label=plot_titles[i_p], args=[{"visible": visible}, {"title": path_out + ': ' + plot_titles[i_p]}], )
    steps.append(step)

sliders = [dict(active=10, currentvalue={"prefix": "Plot: "}, pad={"t": 50}, steps=steps)]
fig.update_layout(title=path_out, sliders=sliders, bargap=0.1, bargroupgap=0.0)
fig.update_xaxes(rangeslider=dict(visible=False))
if histnorm == 'percent':
    fig.update_yaxes(tickmode='array', tickvals=list(range(0, 100, 1)))
fig.write_html(main_path + path_out + '.html', auto_open=auto_open_html)
print(f'Sub plots = {str(plot_counter)[1:-1]};   Total plots = {sum(plot_counter)}')
