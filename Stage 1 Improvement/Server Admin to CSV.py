import os
import sys
import inspect
from datetime import datetime
import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
import arc_th_calc
import log_file
import log_spi_or_scope
from my_pyplot import omit_plot as _O, plot as _P, print_chrome as _PC, clear as _PP, print_lines as _PL, send_mail as _SM
import Plot_Graphs_with_Sliders as _G
import my_tools

folder = r'V:\HW_Infrastructure\Analog_Team\ARC_Data\Results\Jupiter\Jupiter+ Improved (7E0872F4-EC)\Jupiter FAs Vdc 750v vs 850v\CSVs 02 - Holland'
file = '2023.02.14 - 2023.02.27.csv'
df = pd.read_csv(f'{folder}\\{file}', delimiter=',', skiprows=0, keep_default_na=False)
file = '2022.11.01 - 2023.02.13.csv'
df = pd.concat([df, pd.read_csv(f'{folder}\\{file}', delimiter=',', skiprows=0, keep_default_na=False)])

df.drop(df.filter(regex="Unname"), axis=1, inplace=True)
df.drop(['partition_dt', 'physicalreporterid'], axis=1, inplace=True)
df = df[df.duplicated(keep='last', subset=['unit_SN', 'Time']) == False]
df = df[(df['reportertype'] == 3) | (df['reportertype'] == 22)]

df = df.rename(columns={
    'polestarid': 'Portia ID',
    'polestarid_Hex': 'Portia Hex',
    'Time': 'Date',
    'EVENT_STAGE_1': 'Stage 1 Events',
    'EVENT_STAGE_2': 'Stage 2 Events',
    'had_stage_2': 'Had Stage 2',
    'had_stage_1': 'Had Stage 1',
    'location_country': 'Country',
    'reportertype': 'Reporter Type',
    'unit_SN': 'SN',
    'serial_name': 'Inverter',
    'p424': 'Detection Mode',
    'p425': 'Energy TH',
    'P344': 'DC Voltage',
    'p438': 'Current TH',
    'P271': 'AC Voltage',
    'p1240': 'Detection Criteria',
})
df.to_csv(f'{folder}\\{file[:-4]} edit.csv', index=False)
