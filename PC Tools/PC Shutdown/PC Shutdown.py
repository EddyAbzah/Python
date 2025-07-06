"""    #### NOTES ###
To convert to EXE:
    1. open CMD
    2. install pyinstaller → pip install pyinstaller
    3. open source directory
    4. open CMD using the address bar
    5. enter and run:
    pyinstaller --onefile --noconsole "PC Shutdown.py" --icon "icon.png"
    or
    pyinstaller --noconfirm --onedir --windowed --contents-directory "PC Shutdown" --icon "transparent.png"  "PC Shutdown.py"
"""


import os
from tkinter import *       # import all functions with the need for "tkinter."
enable_shutdown = False


def cancel_shutdown():
    global enable_shutdown
    enable_shutdown = False
    if os.name == 'nt':  # For Windows operating system
        os.system(f'shutdown -a')
    else:
        print('Unsupported operating system.')


def calculate_time_string(timer):
    if timer <= 0:
        out_string = f'now'
    elif timer == 1:
        out_string = f'in 1 second'
    elif timer < 60:
        out_string = f'in {timer} seconds'
    elif timer == 60:
        out_string = f'in 1 minute'
    elif timer < 60 ** 2:
        out_string = f'in {timer / 60:0.2f} minutes'
    elif timer == 60 ** 2:
        out_string = f'in 1 hour'
    else:
        out_string = f'in {timer / 60 / 60:0.2f} hours'
    return out_string


def change_label(timer):
    if not enable_shutdown:
        label.config(text='Shutdown is not enabled')
    else:
        label.config(text=f'PC will shutdown {calculate_time_string(timer)}')
        if timer > 0:           # call countdown again after 1000ms (1s)
            root.after(1000, change_label, timer - 1)


def shutdown_computer():
    global enable_shutdown
    cancel_shutdown()
    if os.name == 'nt':     # For Windows operating system
        try:
            timer = float(entry.get())
            multiple = 60
        except ValueError:
            timer = float(clicked.get().split(' ')[0])
            multiple = clicked.get().split(' ')[1]
            if multiple[:6] == "minute":
                multiple = 60
            elif multiple[:4] == "hour":
                multiple = 60 ** 2
            else:
                multiple = 1
        timer *= multiple
        timer = int(timer)
        os.system(f'shutdown /s /t {timer}')
        enable_shutdown = True
        change_label(timer)
    else:
        print('Unsupported operating system.')


root = Tk()                 # Create object
root.geometry("300x300")    # Adjust size
root.title("PC Shutdown")

dropdown_menu_options = ["5 minutes", "10 minutes", "20 minutes", "30 minutes", "45 minutes", "1 hour", "1.5 hours", "2 hours", "3 hours"]

clicked = StringVar()                       # datatype of menu text
clicked.set(dropdown_menu_options[0])       # initial menu text

drop = OptionMenu(root, clicked, *dropdown_menu_options)        # Create Dropdown menu
drop.configure(height=3, width=12, font=30)
drop.pack()

entry = Entry(root, width=5, font=30)
entry.pack()

button_init = Button(root, height=3, width=12, font=30, bg='green', fg='white', text="Set Timer", command=shutdown_computer)         # Create button, it will change label text
button_init.pack()

label = Label(root, height=3, width=45, font=30, text='Shutdown is not enabled')       # Create Label
label.pack()

button_cancel = Button(root, height=3, width=16, font=30, text="Cancel Shutdown", bg='red', fg='white', command=cancel_shutdown)         # Create button, it will change label text
button_cancel.pack()

root.mainloop()         # Execute tkinter
