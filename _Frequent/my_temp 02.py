import plotly.graph_objects as go
import numpy as np
# generate data in DataFrame

data = {
    'x': [1, 2, 3, 4, 5],
    'y_rssi_1': [50, 60, 70, 80, 90],
    'y_snr_1': [1, 1, 1, 1, 1],
    'y_rssi_2': [100, 200, 300, 400, 500],
    'y_snr_2': [2, 2, 2, 2, 2],
}
# Create figure
fig = go.Figure()
# Add traces
scenarios = ['y_rssi_1', 'y_rssi_2']
for scenario in scenarios:
    fig.add_trace(
        go.Scatter(
            visible=(scenario==scenarios[0]),
            line=dict(color="#00ced1", width=6),
            name=scenario,
            x=data['x'],
            y=data[scenario],
            # customdata=np.array([data[scenario.replace('rssi', 'snr')]]*len(data['x'])),
            hovertemplate=
            'x: %{x}<br>' +
            'y: %{y}<br>' +
            'trace: %{name}<extra></extra>'
        )
    )
# Create and add slider
steps = [dict(method='restyle',
              args=['visible', [i==j for i in range(len(scenarios))]],
              label=scenarios[j]) for j in range(len(scenarios))]
fig.update_layout(
    sliders=[dict(
        active=0,
        pad={"t": 50},
        steps=steps
    )]
)
# Create and add button
button_layer_1_height = 1.12
fig.update_layout(
    updatemenus=[
        dict(buttons=list([
            dict(label='Show snr',
                 method='restyle',
                 args=['y', [data[scenario.replace('rssi', 'snr')] for scenario in scenarios]]),
            dict(label='Show rssi',
                 method='restyle',
                 args=['y', [data[scenario] for scenario in scenarios]])
    ]),
            direction="left",
            pad={"r": 10, "t": 10},
            showactive=True,
            x=0.11,
            xanchor="left",
            y=button_layer_1_height,
            yanchor="top"
        ),
    ],
    title_text="Toggle y"
)
# Save figure
fig.write_html("slider_and_button.html", auto_open=True)