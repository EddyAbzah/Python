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
import tkinter as tk
import plotly.offline as pyo
import plotly.graph_objects as go
from datetime import datetime
from tkcalendar import DateEntry
from tkinter.filedialog import asksaveasfilename


# Rehaniya = 33°2′54″N 35°29′15″E
time_zone = "Asia/Jerusalem"
latitude = 33.04848968406511
longitude = 35.48841118810589
max_theoretical_power = 615 * 32
max_inverter_power = 15e3


def calculate_and_plot(selected_date_str, online):
    selected_date = datetime.strptime(selected_date_str, "%Y-%m-%d")
    start_time = pd.Timestamp(selected_date.strftime('%Y-%m-%d'))
    time = pd.date_range(start=start_time, periods=24 * 4, freq='15min')
    time_utc = time.tz_localize(time_zone).tz_convert('UTC')

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
    fig.update_layout(title=f"Solar Production on {selected_date.strftime('%Y-%m-%d')}", xaxis_title="Time", yaxis_title="kW", template="plotly_dark")
    if online:
        fig.show()
    else:
        file_path = asksaveasfilename(defaultextension=".html", filetypes=[("HTML files", "*.html")], title="Save Plot As",
                                      initialfile=f"Solar Production on {selected_date.strftime('%Y-%m-%d')}.html")
        if file_path:
            pyo.plot(fig, filename=file_path, auto_open=True)


def launch_gui():
    root = tk.Tk()
    root.title("Select a Date for Solar Production Graph")
    tk.Label(root, text="Select a date:").pack(pady=5)

    cal = DateEntry(root, width=15, background='black', foreground='white', borderwidth=2, relief='groove', date_pattern='dd-mm-yyyy', selectbackground='orange',
                    font=('Helvetica', 12), selectforeground='black', normalbackground='gray20', normalforeground='white', headersbackground='gray30', headersforeground='lightblue')
    cal.pack(padx=10, pady=10)
    cal.config(justify='center')

    def plot(online):
        selected_date_str = cal.get_date().strftime('%Y-%m-%d')
        calculate_and_plot(selected_date_str, online=online)

    tk.Button(root, text="Plot online", command=lambda: plot(True)).pack(side=tk.LEFT, padx=10, pady=10)
    tk.Button(root, text="Plot offline", command=lambda: plot(False)).pack(side=tk.RIGHT, padx=10, pady=10)
    root.mainloop()


launch_gui()
