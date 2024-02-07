import plotly.graph_objects as go
import pandas as pd

scenarios = ['scenario #1', 'scenario #2']
data_points = ['inv_rssi', 'inv_snr', 'opt_rssi', 'opt_snr']
data = {'x': [1, 2, 3, 4, 5],
        scenarios[0]: {data_points[0]: [60, 60, 70, 80, 80], data_points[1]: [30, 50, 500, 50, 30], data_points[2]: [200, 200, 300, 400, 400], data_points[3]: [200, 150, 100, 150, 200]},
        scenarios[1]: {data_points[0]: [0, 150, 50, 150, 0], data_points[1]: [50, 50, 300, 50, 50], data_points[2]: [200, 200, 230, 200, 230], data_points[3]: [0, 0, 30, 0, 0]}}

# df = pd.DataFrame(data)
fig = go.Figure()
for scenario in scenarios:
    for data_point in data_points:
        fig.add_trace(go.Scatter(visible=(scenario == scenarios[0] and data_point == data_points[0]),
                                 line=dict(width=6), name=f"{scenario} - {data_point}",
                                 x=data['x'], y=data[scenario][data_point]))
steps = list()
for i in range(0, len(fig.data), len(data_points)):
    step = dict(
        method="update",
        args=[{"visible": [False] * len(fig.data)}, {"title": "Slider switched to step: " + str(i)}]
    )
    step["args"][0]["visible"][i:i+len(data_points)] = [True]*len(data_points)
    steps.append(step)
fig.update_layout(
    sliders=[dict(
        active=0,
        currentvalue={"prefix": "Scenario: "},
        pad={"t": 50},
        steps=steps
    )]
)
# Create and add button
def create_button(label, visible_for_all_traces):
    return dict(
        label=label,
        method='update',
        args=[{'visible': [visible_for_all_traces]*len(fig.data)}]
    )

fig.update_layout(
    updatemenus=[
        dict(
            active=0,
            buttons=[
                create_button('Show inv_rssi', False),
                create_button('Show inv_snr', False),
                create_button('Show opt_rssi', False),
                create_button('Show opt_snr', False),
                create_button('Show all', True),
            ]
        )
    ]
)

fig.write_html("slider_and_button.html", auto_open=True)
