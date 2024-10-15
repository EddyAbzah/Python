import pandas as pd
from datetime import datetime

# Paths:
path_folder = r'M:\Users\ShacharB\Projects\PLC Leakage - RSSI Ratio Issue\RSSI Ratio Issue Gen4 - 12.2022\Solution Procedure\Raw Data\New data 14-10-2024'
path_file_in = 'opts_param_129_new2024-02-01_2024-04-30.csv'
# path_file_in = 'opts_param_129_new2024-05-01_2024-07-31.csv'
# path_file_in = 'opts_param_129_new2024-08-01_2024-10-15.csv'
path_file_out = '01.02.2024 - 30.04.2024.csv'
# path_file_out = '01.05.2024 - 31.07.2024.csv'
# path_file_out = '01.08.2024 - 15.10.2024.csv'


def custom_format(x):
    """Custom function to format floats with rounding for smaller numbers"""
    # if not isinstance(x, float):
    #     x = float(x)
    # else:
    if x > 1e6 or x < 1e-6:  # Apply scientific notation for large/small numbers
        return f'{x:.6e}'  # One digit before the decimal point in scientific notation
    else:
        return str(round(x))  # Round the number and convert to string


# Get DFs:
df = pd.read_csv(f'{path_folder}\\{path_file_in}')
print("\nDF before editing:")
print(df.head(3).to_string())

# Set the data like this:
# deviceid,managerid,optimizerid,param_129,optimizerid_hex,time
# "M:\Users\ShacharB\Projects\PLC Leakage - RSSI Ratio Issue\RSSI Ratio Issue Gen4 - 12.2022\Solution Procedure\236 Day Analysis\RSSI_Issue_-_P129_no_duplicates_2024_02_22.csv"
df["optimizerid_hex"] = df["optimizerid"].apply(lambda x: hex(int(x))[2:].upper())  # [2:] to remove the 0x prefix
if "date_updated" in df.columns.tolist():
    del df["date_updated"]
df["time"] = df["time"].apply(lambda x: datetime.strptime(x, "%d/%m/%Y %H:%M:%S").strftime("%Y-%m-%d %H:%M:%S.%f")[:-3] + " UTC")

# Apply the custom formatting function to the DataFrame instead of float_format='%.6e' inside df.to_csv():
df["param_129"] = df["param_129"].apply(custom_format)

# Sort the columns:
df = df[["deviceid", "managerid", "optimizerid", "param_129", "optimizerid_hex", "time"]]

# Save the data:
print("\nDF after editing:")
print(df.head(3).to_string())
df.to_csv(f'{path_folder}\\{path_file_out}', index=False)
