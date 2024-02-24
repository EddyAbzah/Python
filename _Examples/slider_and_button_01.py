import plotly.graph_objects as go
import numpy as np

scenarios = ['Scenario #1', 'Scenario #2']
data = {'x': [1, 2, 3, 4, 5], scenarios[0]: {'inv_rssi': [60, 60, 70, 80, 80], 'inv_snr': [30, 50, 50, 50, 30], 'opt_rssi': [200, 200, 300, 400, 400], 'opt_snr': [200, 150, 100, 150, 200]},
        scenarios[1]: {'inv_rssi': [0, 150, 50, 150, 0], 'inv_snr': [0, 0, 300, 0, 0], 'opt_rssi': [0, 0, 30, 0, 30], 'opt_snr': [0, 0, 30, 0, 0]}}
fig = go.Figure()

for scenario in scenarios:
    fig.add_trace(go.Scatter(visible=(scenario == scenarios[0]), line=dict(color="#00ced1", width=6), name=scenario, x=data['x'], y=data[scenario]['inv_rssi'],
                             hovertemplate=f'{scenario}<br>' + 'x: %{x}<br>' + 'y: %{y:.2f}<extra></extra>'))

steps = [dict(method='restyle', args=['visible', [i == j for i in range(len(scenarios))]], label=scenarios[j]) for j in range(len(scenarios))]
fig.update_layout(sliders=[dict(active=0, pad={"t": 50}, steps=steps)])
fig.update_layout(updatemenus=[dict(buttons=list([dict(label='Show inv_rssi', method='restyle', args=['y', [data[scenario]['inv_rssi'] for scenario in scenarios]]),
                                                  dict(label='Show inv_snr', method='restyle', args=['y', [data[scenario]['inv_snr'] for scenario in scenarios]]),
                                                  dict(label='Show opt_rssi', method='restyle', args=['y', [data[scenario]['opt_rssi'] for scenario in scenarios]]),
                                                  dict(label='Show opt_snr', method='restyle', args=['y', [data[scenario]['opt_snr'] for scenario in scenarios]]),
                                                  dict(label='Show all', method='restyle', args=['y', [list(data[scenario].values()) for scenario in scenarios]])]),
                                    direction="right", pad={"r": 10, "t": 10}, showactive=True, x=0.11, xanchor="right", y=1.12, yanchor="top"),])

# fig.write_html("slider_and_button.html", auto_open=True)
fig.show()
