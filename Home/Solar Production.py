"""
---------------------------------------------------------
Title: Compare solar panel energy production to theory
Description: Inspired by this tutorial: https://www.youtube.com/watch?v=OHxR8iMHDWw
             Converted from MATLAB to Python by me
---------------------------------------------------------
"""


import pvlib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from my_pyplot import omit_plot as _O, plot as _P, print_chrome as _PC, clear as _PP, print_lines as _PL, send_mail as _SM


# Rehaniya = 33°2′54″N 35°29′15″E
time_zone = "Asia/Jerusalem"
latitude = 33.04848968406511
longitude = 35.48841118810589
max_theoretical_power = 615 * 32
max_inverter_power = 15e3


time = pd.date_range('2025-07-22', periods=24 * 4, freq='15min')
time_utc = time.tz_localize('Asia/Jerusalem').tz_convert('UTC')
solar_position = pvlib.solarposition.get_solarposition(time_utc, latitude, longitude)
solar_position.index = time

declination = solar_position['apparent_elevation']
sun_angle = solar_position['apparent_elevation'].apply(lambda x: np.sin(np.radians(x)))
sun_intensity = sun_angle.apply(lambda x: 1.4883 * 0.7 ** (x ** -0.678) if x > 0 else 0)
production_theory = max_theoretical_power * sun_intensity * sun_angle
production_expected = production_theory.where(production_theory <= max_inverter_power, max_inverter_power)

fig = go.Figure()
fig.add_trace(go.Scatter(x=production_theory.index, y=production_theory, mode='lines+markers', name='Production Theory'))
fig.add_trace(go.Scatter(x=production_expected.index, y=production_expected, mode='lines+markers', name='Production Expected'))
fig.update_layout(title="Compare solar panel energy production to theory", xaxis_title="Time", yaxis_title="kW", template="plotly_dark")
fig.show()
