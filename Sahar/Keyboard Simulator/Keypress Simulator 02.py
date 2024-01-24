# How to create an EXE file:
# pyinstaller --onefile --noconsole --icon "icon.ico" "Keypress Simulator 03.py"
# pyinstaller --noconfirm --onedir --windowed --icon "icon.ico"  "Keypress Simulator 03.py"

import sys
import time
from tkinter import Tk, IntVar, Checkbutton, Text
from pynput import keyboard

stop_app = False


# close the GUI:
def close():
    global stop_app
    stop_app = True
    listener.stop()
    root.destroy()
    sys.exit()


# filter out Paste event:
def win32_event_filter(msg, data):
    if suppress_paste_events.get() and (msg == 256 or msg == 257) and data.vkCode == 86:
        listener._suppress = True
    else:
        listener._suppress = False
    return True


def on_press(key):
    # close the GUI:
    if key == keyboard.Key.esc:
        close()
    # check if Paste:
    elif type(key) is type(keyboard.KeyCode()) and '\\' in repr(key) and ord(key.char) == 22:
        try:
            text = Tk().clipboard_get()
            if not check_if_string.get() or isinstance(text, str):
                if replace_new_lines.get():
                    text = text.replace('\n', textbox.get(1.0, "end-1c")).replace('\r', '').strip()
                time.sleep(0.8)
                keyboard.Controller().type(text)
        except:
            pass


# hide / show textbox in GUI:
def disable_text():
    if replace_new_lines.get():
        textbox.pack()
    else:
        textbox.pack_forget()


with keyboard.Listener(on_press=on_press, win32_event_filter=win32_event_filter, suppress=False) as listener:
    root = Tk()
    root.geometry("500x260")
    root.title("Keyboard Simulator")

    # Bool button 01:
    suppress_paste_events = IntVar()
    suppress_paste_events.set(True)
    Checkbutton(root, text='Suppress Paste events', font="arial 30", variable=suppress_paste_events, onvalue=1, offvalue=0).pack()

    # Bool button 02:
    check_if_string = IntVar()
    check_if_string.set(True)
    Checkbutton(root, text='Paste only if String', font="arial 30", variable=check_if_string, onvalue=1, offvalue=0).pack()

    # Bool button 03:
    replace_new_lines = IntVar()
    replace_new_lines.set(True)
    Checkbutton(root, text='Replace New Lines', font="arial 30", variable=replace_new_lines, onvalue=1, offvalue=0, command=disable_text).pack()

    # Text input:
    textbox = Text(root, font="arial 30", height=1, width=20)
    textbox.pack()

    root.protocol("WM_DELETE_WINDOW", close)
    root.mainloop()
    if not stop_app:
        listener.join()
