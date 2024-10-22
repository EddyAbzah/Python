import os
import pandas as pd
from datetime import datetime

# Paths:
path_folder = r'C:\Users\eddy.a\Downloads\RSSI Ratio - Meyer Burger modules\Monitoring'
path_file_out = 'Power + Weather.csv'
key_words = ["Power", "Weather"]


temp_df = pd.DataFrame()
merged_dfs = []
for file in [f for f in os.listdir(path_folder) if f.endswith(".csv") and f != path_file_out]:
    if any(item in file for item in key_words):
        if temp_df.empty:
            temp_df = pd.read_csv(os.path.join(path_folder, file)).dropna(how='all', axis='columns')
        else:
            temp_df = pd.merge(temp_df, pd.read_csv(os.path.join(path_folder, file)).dropna(how='all', axis='columns'), on='Time')
            temp_df["Site ID"] = file.split(' _ ')[0]
            merged_dfs.append(temp_df)
            temp_df = pd.DataFrame()
    else:
        print(f"There is no keyword match for {file = }")

df_full = pd.concat(merged_dfs, ignore_index=True)
df_full.columns = ["Date", "AC Production (W)", "AC Consumption (W)", "Humidity (%)", "Temperature (C)", "Site ID"]
df_full = df_full[["Date", "Site ID", "AC Production (W)", "AC Consumption (W)", "Humidity (%)", "Temperature (C)"]]

# Save the data:
print("\nDF after editing:")
print(df_full.head(3).to_string())
df_full.to_csv(f'{path_folder}\\{path_file_out}', index=False)
