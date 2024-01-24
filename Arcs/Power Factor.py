import numpy as np
import pandas as pd
from my_pyplot import plot as _P, print_chrome as _PC, clear as _PP, print_lines as _PL, send_mail as _SM
import Plot_Graphs_with_Sliders as _G
import my_tools

# Data:
path_in = r'M:\Users\LinoyL\Vdc_Criteria_new\MOHAMD_ARCS\RAW_DATA\S1\A1\S1_InvCb_7A\S1_7 Rec003 spi 7[A] x3Str Arcer1_InvToCB.txt'
df = pd.read_csv(path_in).dropna(how='all', axis='columns')
iac_wave = np.array(df['Iac_L1,'])
vac_wave = np.array(df['Vac_L1'])
sampling_frequency = 16667

# Normalize the waves:
iac_wave = (iac_wave - np.mean(iac_wave)) / np.std(iac_wave)
vac_wave = (vac_wave - np.mean(vac_wave)) / np.std(vac_wave)

# Cross-correlation:
xcorr = np.correlate(iac_wave, vac_wave, 'full')

# Delta index which maximizes the cross-correlation is the shift that brings the two waves into best alignment:
delta_index = np.argmax(xcorr) - len(iac_wave) + 1

# Calculate phase difference (phi):
phi = 2.0 * np.pi * delta_index / sampling_frequency

# Power Factor is cosine of the phase difference
power_factor = np.cos(phi)

print('Phase difference (in radians):', phi)
print('Power factor:', power_factor)
