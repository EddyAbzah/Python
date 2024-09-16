# ## This will read the CSV as text, because reading it as float is not perfect ## # 

import pandas as pd
import numpy as np

folder = r'C:\Users\eddy.a\Downloads\Mixer with LT Spice\Tests 03'
file_in = 'Scope _ 001 - Lrx.csv'
file_out = 'Scope _ 001 - Lrx_.csv'
delimiter = ','
time_start = 0
time_end = 10

df = pd.read_csv(folder + '\\' + file_in, delimiter=delimiter, dtype=object)
time = np.linspace(time_start, time_end, len(df))
time = pd.DataFrame([f'{t:.6f}' for t in time], dtype=object)
df = pd.concat([time, df], ignore_index=True, axis=1)
df = df.drop(df.columns[1], axis=1)
print(f'Creating CSV of shape = {df.shape}')
df.to_csv(folder + '\\' + file_out, header=False, index=False)
