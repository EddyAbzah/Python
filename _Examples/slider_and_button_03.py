import plotly.graph_objects as go
import pandas as pd


path_folder = r'M:\Users\HW Infrastructure\PLC team\INVs\Jupiter48\Jupiter48 BU - New layout + DC conducted - EddyA 2.2024\Cable Automation\Cable Automation 05'
path_file_out = 'Jup48 New DC Filter'
df = pd.read_csv(f'{path_folder}\\{path_file_out}.csv')
scenarios = [s for s in df.iloc[:, :10].fillna('off').astype(str).agg(', '.join, axis=1).unique()]
data_points = df['Measurement'].unique()

set_yaxis_range = [False, 0, 45]
set_summary_traces = [True, 4]
remove_nans_from_plot = True


sex = [pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()]


fig = go.Figure()
plot_counter = []
for scenario_index, scenario in enumerate(scenarios):
    scenario = scenario.split(' = ')[0]
    plot_count = []
    for data_point_index, data_point in enumerate(data_points):
        print(', '.join([f'{scenario_index = }   {data_point_index = }']))
        sdf = df[df['Measurement'] == data_point]
        sdf = sdf[sdf['Scenario'].str.contains(scenario + ' = ')]
        sdf = sdf.iloc[:, 11:]
        plot_count.append(len(sdf))
        sex[data_point_index] = pd.concat([sex[data_point_index], sdf], axis=0)
        for row in sdf.iterrows():
            line_dict = None
            plot_name = f'{data_point} - {row[1][0]}'
            if set_summary_traces[0] and 'Sample' not in row[1][0]:
                line_dict = dict(width=set_summary_traces[1])
            if remove_nans_from_plot:
                row = row[1].dropna()[1:]
            else:
                row = row[1][1:]
            fig.add_trace(go.Scatter(visible=(scenario_index == 0 and data_point_index == 0), name=plot_name, y=list(row), x=row.keys(), line=line_dict,
                          hovertemplate=f'{scenario}<br>' + 'x: %{x}<br>' + 'y: %{y:.2f}<extra></extra>'))
            if set_yaxis_range[0]:
                fig.update_layout(yaxis_range=[set_yaxis_range[1], set_yaxis_range[2]])
    plot_counter.append(plot_count)
fig.update_layout(title=scenarios[0], title_font_color="#407294", title_font_size=30, legend_title="Traces:", legend_title_font_color="green")


steps = list()
slider_counter = [sum(s) for s in plot_counter]
for i_trace, traces in enumerate(slider_counter):
    step = dict(label=scenarios[i_trace].split(' = ')[0], method="update", args=[{"visible": [False] * len(fig.data)}, {"title": scenarios[i_trace]}])
    step["args"][0]["visible"][sum(slider_counter[:i_trace]):sum(slider_counter[:i_trace]) + traces] = [True] * traces
    steps.append(step)
fig.update_layout(sliders=[dict(active=0, pad={"t": 50}, steps=steps)])


def create_button(label, visible):
    if visible == 0:
        visible = [True] * len(fig.data)
    else:
        visible = [[visible == 1], [visible == 2], [visible == 3], [visible == 4]]
        visible = [[a * b for a, b in zip(visible, pc)] for pc in plot_counter]
        visible = [c for b in [c for b in visible for c in b] for c in b]
    return dict(label=label, method='restyle', args=[{'visible': visible}])


# fig.update_layout(updatemenus=[dict(active=0, buttons=[
#     create_button('Inverter SNR', 1),
#     create_button('Inverter RSSI', 2),
#     create_button('Optimizer SNR', 3),
#     create_button('Optimizer RSSI', 4),
#     create_button('All', 0)])])

fig.update_layout(updatemenus=[dict(buttons=list([
    dict(label='Inverter SNR', method='restyle', args=['y', sex[0]]),
    dict(label='Inverter RSSI', method='restyle', args=['y', sex[1]]),
    dict(label='Optimizer SNR', method='restyle', args=['y', sex[2]]),
    dict(label='Optimizer RSSI', method='restyle', args=['y', sex[3]])]))])
    # dict(label='All', method='restyle', args=['y', [list(data[scenario].values()) for scenario in scenarios]])])


fig.write_html(path_folder + "\\slider_and_button.html", auto_open=True)
