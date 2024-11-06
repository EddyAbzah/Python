import pandas as pd


file_in = r"C:\Users\eddy.a\Downloads\RSSI Ratio - Meyer Burger modules\RSSI_Ratio_Meyer_Burger_modules_04112024.csv"
file_out = r"C:\Users\eddy.a\Downloads\RSSI Ratio - Meyer Burger modules\Ratio_Meyer_Burger_modules_2024_11_04.csv"

df = pd.read_csv(file_in)
if "Unnamed: 0" in df:
    df.drop(df.columns[0], axis=1, inplace=True)
df['get_time'] = pd.to_datetime(df['get_time']).dt.strftime('%d/%m/%Y %H:%M')
df.to_csv(r"C:\Users\eddy.a\Downloads\RSSI Ratio - Meyer Burger modules\RSSI_Ratio_Meyer_Burger_modules_04112024.csv", index=False)