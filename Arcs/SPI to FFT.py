import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import pandas as pd
from typing import Tuple
from my_pyplot import plot as _P, print_chrome as _PC, clear as _PP, print_lines as _PL, send_mail as _SM
import Plot_Graphs_with_Sliders as _G
import my_tools


def fft(s_time: np.ndarray, n_samples: int) -> Tuple[np.ndarray]:  # FFT process
    s_time_t = s_time.T
    s_fft = sp.fft.fft(s_time_t)
    s_fft_abs = np.abs(s_fft / n_samples).T
    # FFT - taking only half of the data [fft output only contain half of "real" data,
    # the other half is just a mirror]
    s_hs = s_fft_abs[0:int((n_samples / 2) + 1)]
    # Throwing bit 0 (dc component), also multipling by 2 , because we remove half the data
    # then we will need to divide bt sqrt(2) in order to get rms
    s_hs[1:int(n_samples / 2 + 1)] = 2 * s_hs[1:int(n_samples / 2 + 1)]
    return s_hs


csv_path_folder = r"M:\Users\Eddy A\Orion\04 Orion Lab F3 (E08EFB51)\Automation\Scope vs Spectrum comparison v5\\"
csv_file_name = "With Telems 12kW No test equipment"
csv_path_input_file = csv_file_name + ".txt"
csv_path = csv_path_folder + csv_path_input_file
# number of samples used for the FFT
# Fsample = 2e6 , Nsamples = 2^14 -> meaning Fresoultion = 2e6/2^12 ~122.07Hz
#n_samples = 2 ** 14
# number of arrays used for to average the FFT
n_ffts = 2 ** 4  # 2**4 = 16 the same as asic matlab ; 2**8 = 256
n_samples = 128
csv_path_out_file_avg = csv_path_folder + csv_file_name + "_fft_n_samples=" + "_AVG=" + str(
    n_ffts) + ".csv"
csv_path_out_file_max = csv_path_folder + csv_file_name + "_fft_n_samples="  + "_MAX=" + str(
    n_ffts) + ".csv"
# ## CSV read return the vector is data frame
# ## csv_path - path to the csv file
# ## nrows - number of rows taken out o f the CSV/Record
# ## usecols=[0] - meaning less , but in case data frame is a matrix, you can control
# ## how many columes you want to take out of the CSV
# ## astype(float) - return the data in the data frame as float
s_df = pd.read_csv(csv_path, usecols=[0]).astype(float)
# Data frame is being cast to array
s = s_df.to_numpy().flatten()

fs = 40e3  # Sampling frequency
T = 1 / fs  # Sampling period
adc_bits = 14  # ADC number of bits
adc_res = 2 ** adc_bits  # ADC resolution
adc_ref = 2.5  # ADC reference voltage

n_samples=int(len(s_df)/n_ffts)
s = s[:n_ffts * n_samples]

s_data = np.reshape(s, newshape=(n_ffts, n_samples)).T
# empty_s = np.zeros((n_samples,n_ffts), dtype=int)
empty_s = np.zeros((int(n_samples / 2) + 1, n_ffts))
for i in range(n_ffts):
    print(i, n_ffts)
    s_array = s_data[:, [i]]
    s_data2 = fft(s_array, n_samples)
    # s_data2 = fft(s_data[:,[i]],n_samples)
    empty_s[:, [i]] = s_data2
    # empty_s[:,[i]] = s_data2
s_avg = np.sum(empty_s, axis=1) / n_ffts
s_max = np.max(empty_s, axis=1)

# creating x-axis that start from 0(DC) to fs/2
f = fs / n_samples * np.linspace(start=0, stop=int(n_samples / 2) - 1, num=int(n_samples / 2) + 1)

s_avg_vrms = (s_avg * adc_ref) / (adc_res)
s_avg_dbm = 10 * np.log10(((s_avg_vrms ** 2) / 50) * 1000)

s_max_vrms = (s_max * adc_ref) / (adc_res)
s_max_dbm = 10 * np.log10(((s_max_vrms ** 2) / 50) * 1000)

pd.DataFrame(s_avg_dbm, f).to_csv(csv_path_out_file_avg)
pd.DataFrame(s_max_dbm, f).to_csv(csv_path_out_file_max)


plt.subplot(121)
plt.plot(f, s_avg_dbm)
plt.title('AVG Spectrum of S(t)')
plt.xlabel('f (Hz)')
plt.ylabel('|s(f)|[dBm]')
plt.grid()
plt.show()

plt.subplot(122)
plt.plot(f, s_max_dbm)
plt.title('Max Hold Spectrum of S(t)')
plt.xlabel('f (Hz)')
plt.ylabel('|s(f)|[dBm]')
plt.grid()
plt.show()
