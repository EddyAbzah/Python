# pyinstaller --onefile main.py
import tkinter as tk
from tkinter import filedialog as tkFileDialog
import pandas as pd
from tkinter import messagebox as tkMessageBox


def browse_file():
    filename = tkFileDialog.askopenfilename()
    entry_file.delete(0, tk.END)
    entry_file.insert(0, filename)
    entry_resample.get()

def convert_to_volts():
    df = pd.read_csv(entry_file.get(), delimiter='[,\t]')
    print(df)


def convert_to_dbs():
    with open('C:/path/numbers.txt') as f:
        lines = f.read().splitlines()


def resample():
    df_ltspice = pd.read_csv(StringIO(ltspice_file), delimiter=',', skiprows=1)
    x = list(df_ltspice.iloc[:, 0].squeeze())
    y = df_ltspice.iloc[:, index_df].squeeze()
    f = scipy.interpolate.interp1d(x, y)
    x = list(df.iloc[:, 0].squeeze())
    y = f(x)

window = tk.Tk()
window.title('LTspice Conversion Tool')
# window.geometry('500x350')

label_main = tk.Label(text='LTspice Conversion Tool', fg='red', font=('helvetica', 18, 'bold')).grid(row=0, column=0, padx=50, pady=20)
label_file = tk.Label(text='Input file (.csv or .txt):').grid(row=1, column=0, sticky=tk.W, padx=5, pady=1)
entry_file = tk.Entry(width=55)
entry_file.grid(row=2, column=0, sticky=tk.W, padx=5, pady=1)
button_file = tk.Button(text='Browse', command=browse_file).grid(row=2, column=0, sticky=tk.E, padx=5, pady=1)
label_resample = tk.Label(text='Resample file (put 0 to skip):').grid(row=3, column=0, sticky=tk.W, padx=5, pady=1)
entry_resample = tk.Entry(width=10)
entry_resample.grid(row=4, column=0, sticky=tk.W, padx=5, pady=1)
label_space = tk.Label(font=('helvetica', 10, 'bold')).grid(row=5, column=0, padx=50, pady=5)
button_volt = tk.Button(text='Convert to Volts', command=convert_to_volts).grid(row=6, column=0, sticky=tk.W, padx=50, pady=20)
button_dbm = tk.Button(text='Convert to dBm', command=convert_to_dbs).grid(row=6, column=0, sticky=tk.E, padx=50, pady=20)

window.mainloop()   # code stops here; line after this will run after you close the window
