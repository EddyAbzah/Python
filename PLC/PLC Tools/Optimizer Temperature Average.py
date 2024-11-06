import os
import re
import sys
import fnmatch
import pandas as pd
if os.getlogin() == "eddy.a":
    from my_pyplot import omit_plot as _O, plot as _P, print_chrome as _PC, clear as _PP, print_lines as _PL, send_mail as _SM
    import Plot_Graphs_with_Sliders as _G
    import my_tools


def read_optimizers(file, file_suffix):
    df = pd.read_excel(file)
    df = df[['SN', 'TemperatureC', 'TELEM DATE', 'TELEM TIME']]
    df['DateTime'] = pd.to_datetime(df['TELEM DATE'] + ' ' + df['TELEM TIME'], format="%d/%m/%Y %H:%M:%S")
    df = df.set_index('DateTime')
    df = df.drop(columns=['TELEM DATE', 'TELEM TIME'])
    df = df.rename(columns={"SN": "Optimizer ID", "TemperatureC": "Temperature [°C]"})

    df_pivoted = df.pivot(columns="Optimizer ID", values="Temperature [°C]")
    df_pivoted = df_pivoted.resample('30T').mean()
    df_pivoted.dropna(how='all', inplace=True)              # Drop rows that are completely filled with NaN values
    df_pivoted = df_pivoted.interpolate(method='time')
    df_pivoted = df_pivoted.fillna(method='bfill').fillna(method='ffill')   # bfill = backward fill; ffill = forward fill
    df_pivoted['Average'] = df_pivoted.mean(axis=1)
    df_pivoted.index.strftime('%d/%m/%Y %H:%M')
    df_pivoted = df_pivoted.round(2)

    df_std = df_pivoted.std()
    df_std.sort_values(ascending=False, inplace=True)
    print("Standard deviation for each Optimizer:\n")
    print(df_std)

    file_output = os.path.splitext(file)[0] + file_suffix
    df_pivoted.to_csv(file_output + ".csv")
    _PC(df_pivoted, path=os.path.dirname(file), file_name=os.path.basename(file_output), title=file_out, auto_open=False)


if __name__ == "__main__":
    T_read_optimizers__F_combine_to_site = False
    output_text = False
    output_text_path = r"M:\Users\ShacharB\Projects\PLC Leakage - RSSI Ratio Issue\RSSI Ratio Issue Gen4 - 12.2022\Solution Procedure\Analysis from 30-08-2023 to 15-10-2024\Python log.txt"

    folder = r"M:\Users\HW Infrastructure\PLC team\ARC\Temp-Eddy\RSSI Ratio - Meyer Burger modules\Telems"
    # file_filter = "ServerAdmin*.xlsx"
    file_filter = "ServerAdmin*.csv"
    file_out_suffix = " " + "Pivot (04-11-2024)"
    file_out = "Optimizer Temperatures (04-11-2024)"

    if output_text:
        default_stdout = sys.stdout
        sys.stdout = open(output_text_path, 'w')

    if not T_read_optimizers__F_combine_to_site:
        all_sites = []

    for filename in os.listdir(folder):
        if fnmatch.fnmatch(filename, file_filter):
            file_path = os.path.join(folder, filename)
            print(f"{file_path = }")
            if T_read_optimizers__F_combine_to_site:
                read_optimizers(file_path, file_suffix=file_out_suffix)
            else:
                df = pd.read_csv(file_path, index_col=0)
                df = df["Average"]
                site_name = re.search(r'(\b[A-F0-9]{8}\b)', os.path.basename(file_path)).group(0)
                df.rename(site_name, inplace=True)
                all_sites.append(df)

    if not T_read_optimizers__F_combine_to_site:
        df = pd.concat(all_sites, axis=1)
        df.sort_index(axis=0, ascending=True, inplace=True)
        df = df.interpolate()
        df = df.bfill().ffill()     # bfill = backward fill; ffill = forward fill
        df.to_csv(os.path.join(folder, file_out) + ".csv")
        _PC(df, path=folder, file_name=file_out, title=file_out, auto_open=False)

    if output_text:
        sys.stdout.close()
        sys.stdout = default_stdout
