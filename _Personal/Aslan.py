import os
import time
import ctypes
import pandas as pd
import datetime as dt
from my_pyplot import omit_plot as _O, plot as _P, print_chrome as _PC, clear as _PP, print_lines as _PL, send_mail as _SM
import Plot_Graphs_with_Sliders as _G
import my_tools

# ####   True   ###   False   #### #
output_csv_T__print_F = False
sub_script = 1
Device = ['Black Kodak', 'Toshiba', 'SeaGate', 'WD Black', 'WD Red'][4]
run = ['List files', 'Find files'][sub_script]
search_path = [r'D:\\', r'C:\Users\eddy.a\Downloads\Aslan\Search 02 - USBs only (03-11-2023)'][sub_script]
search_output = r'C:\Users\eddy.a\Downloads\Aslan\Search 02 - USBs only (03-11-2023)' + '\\' + Device + ' - ' + ['Search files.csv', 'Find files.csv'][sub_script]
search_drives = ['Asus new.txt', 'Asus old - C.txt', 'Asus old - D.txt', 'Note 8.txt', 'SeaGate.txt', 'WD Black.txt', 'WD Red.txt']
search_drives = ['Black Kodak - Search files.csv', 'SeaGate - Search files.csv', 'Toshiba - Search files.csv', 'WD Red - Search files.csv']
search_formats = ['.opus', '.mp3', '.mp4', '.m4a', '.ogg', '.wav', '.flac', '.aac']
search_dates = ['20/04/2023', '21/04/2023', '22/04/2023', '23/04/2023', '24/04/2023',
                '04/20/2023', '04/21/2023', '04/22/2023', '04/23/2023', '04/24/2023']
skip_if_recycle_bin = False


all_data = []
if run == 'List files':
    if Device == 'Toshiba':
        search_path += 'KDAFI'
    for root, dirs, files in os.walk(search_path):
        if 'RECYCLE.BIN' in root and skip_if_recycle_bin:
            print(f'Jumping over root = {root}')
            continue
        for file_name in files:
            file_path = os.path.join(root, file_name)
            file_size = os.path.getsize(file_path) / (1024 ** 2)
            all_data.append({
                "Folder": root,
                "File name": file_name,
                "Size (MB)": round(file_size, 2),
                "Date created": dt.datetime.fromtimestamp(os.path.getctime(file_path)).strftime("%d/%m/%Y"),
                "Time created": dt.datetime.fromtimestamp(os.path.getctime(file_path)).strftime("%H:%M:%S"),
                "Date modified": dt.datetime.fromtimestamp(os.path.getmtime(file_path)).strftime("%d/%m/%Y"),
                "Time modified": dt.datetime.fromtimestamp(os.path.getmtime(file_path)).strftime("%H:%M:%S"),
                "Hidden": ctypes.windll.kernel32.GetFileAttributesW(str(file_path)) == -1 or bool(ctypes.windll.kernel32.GetFileAttributesW(str(file_path)) & 0x2)
            })


elif run == 'Find files':
    for file in search_drives:
        with open(search_path + '\\' + file, 'r', errors='replace') as f:
            lines = f.readlines()
            for index, line in enumerate(lines):
                if any([s in line for s in search_formats]) and any([s in line for s in search_dates]):
                    all_data.append({
                        "File": file,
                        "Index": index,
                        "Line": line.rstrip()
                    })


df = pd.DataFrame(all_data)
if output_csv_T__print_F:
    df.to_csv(search_output, index=False)
else:
    print(df.to_string())
