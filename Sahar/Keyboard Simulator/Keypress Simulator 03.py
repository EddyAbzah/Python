# How to create an EXE file:
# pyinstaller --onefile --noconsole --icon "icon.ico" "Keypress Simulator 03.py"
# pyinstaller --noconfirm --onedir --windowed --contents-directory "Keypress Simulator 03" --icon "icon.ico"  "Keypress Simulator 03.py"

import sys
from tkinter import Tk, IntVar, Checkbutton, Text, Button
from pynput import keyboard

stop_app = False
enable_application = True
ctrl_pressed = False


# close the GUI:
def close():
    global stop_app
    stop_app = True
    listener.stop()
    root.destroy()
    sys.exit()


# filter out Paste event:
def win32_event_filter(msg, data):
    if enable_application and suppress_paste_events.get() and ctrl_pressed and (msg == 256 or msg == 257) and data.vkCode == 86:
        listener._suppress = True
    else:
        listener._suppress = False
    return True


def on_release(key):
    global ctrl_pressed
    if key == keyboard.Key.ctrl_l or key == keyboard.Key.ctrl_r:
        ctrl_pressed = False


def on_press(key):
    global ctrl_pressed
    if key == keyboard.Key.ctrl_l or key == keyboard.Key.ctrl_r:
        ctrl_pressed = True
    if key == keyboard.Key.esc:
        listener.stop()
        close()     # close the GUI:
    elif enable_application and type(key) is type(keyboard.KeyCode()) and '\\' in repr(key) and ord(key.char) == 22:        # check if Paste:
        try:
            text = Tk().clipboard_get()
            if not check_if_string.get() or isinstance(text, str):
                if replace_new_lines.get():
                    text = text.replace('\n', textbox.get(1.0, "end-1c")).replace('\r', '').strip()
                ctrl_pressed = False
                keyboard.Controller.release(keyboard.Controller(), key=keyboard.Key.ctrl_l)
                keyboard.Controller.release(keyboard.Controller(), key=keyboard.Key.ctrl_r)
                keyboard.Controller().type(text)
        except:
            pass


def disable_text():     # hide / show textbox in GUI:
    if replace_new_lines.get():
        textbox.pack()
    else:
        textbox.pack_forget()


def set_application():
    global enable_application
    enable_application = not enable_application
    main_button["text"] = ("Disable" if enable_application else "Enable") + " application"
    main_button["bg"] = "red" if enable_application else "green"


def set_always_on_top():
    root.attributes('-topmost', always_on_top.get())


with (keyboard.Listener(on_press=on_press, on_release=on_release, win32_event_filter=win32_event_filter, suppress=False) as listener):
    root = Tk()
    root.geometry("280x220")
    root.title("Keyboard Simulator")
    font = "arial 15"

    # Main button:
    main_button = Button(root, text=("Disable" if enable_application else "Enable") + " application", bg='red', font=font, command=set_application)
    main_button.pack()

    # Bool button 01:
    always_on_top = IntVar()
    always_on_top.set(True)
    Checkbutton(root, text='Always on top', font=font, variable=always_on_top, onvalue=1, offvalue=0, command=set_always_on_top).pack()
    root.attributes('-topmost', always_on_top.get())

    # Bool button 02:
    suppress_paste_events = IntVar()
    suppress_paste_events.set(True)
    Checkbutton(root, text='Suppress Paste events', font=font, variable=suppress_paste_events, onvalue=1, offvalue=0).pack()

    # Bool button 03:
    check_if_string = IntVar()
    check_if_string.set(True)
    Checkbutton(root, text='Paste only if String', font=font, variable=check_if_string, onvalue=1, offvalue=0).pack()

    # Bool button 04:
    replace_new_lines = IntVar()
    replace_new_lines.set(True)
    Checkbutton(root, text='Replace New Lines', font=font, variable=replace_new_lines, onvalue=1, offvalue=0, command=disable_text).pack()

    # Text input:
    textbox = Text(root, font=font, height=1, width=20)
    textbox.pack()

    root.protocol("WM_DELETE_WINDOW", close)
    root.mainloop()
    if not stop_app:
        listener.join()
