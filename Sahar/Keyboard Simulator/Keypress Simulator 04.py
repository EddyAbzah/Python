# How to create an EXE file:
# pyinstaller --onefile --noconsole --icon "icon.ico" "Keypress Simulator 04.py"
# pyinstaller --noconfirm --onedir --windowed --contents-directory "Keypress Simulator 04" --icon "icon.ico"  "Keypress Simulator 04.py"

import sys
from tkinter import Tk, IntVar, Checkbutton, Text, Button
from pynput import keyboard

stop_app = False
enable_application = True
ctrl_pressed = False
replace_new_lines_with = '; '


# close the GUI:
def close():
    global stop_app
    stop_app = True
    listener.stop()
    root.destroy()
    sys.exit()


# filter out Paste event:
def win32_event_filter(msg, data):
    if enable_application and ctrl_pressed and (msg == 256 or msg == 257) and data.vkCode == 86:
        listener._suppress = True
    else:
        listener._suppress = False
    return True


def on_release(key):
    global ctrl_pressed
    if key == keyboard.Key.ctrl_l or key == keyboard.Key.ctrl_r:
        ctrl_pressed = False


def on_press(key):
    global enable_application
    global ctrl_pressed
    if key == keyboard.Key.ctrl_l or key == keyboard.Key.ctrl_r:
        ctrl_pressed = True
    if key == keyboard.Key.esc:
        listener.stop()
        close()     # close the GUI:
    elif enable_application and type(key) is type(keyboard.KeyCode()) and '\\' in repr(key) and ord(key.char) == 22:        # check if Paste:
        try:    # this will fail if the item in the clipboard is not text
            text = Tk().clipboard_get()
            ctrl_pressed = False
            keyboard.Controller.release(keyboard.Controller(), key=keyboard.Key.ctrl_l)
            keyboard.Controller.release(keyboard.Controller(), key=keyboard.Key.ctrl_r)
            if custom_replace_new_lines.get():
                text = text.replace('\n', textbox.get(1.0, "end-1c")).replace('\r', '').strip()
                keyboard.Controller().type(text)
            else:
                text = text.split('\n')
                for line in text:
                    keyboard.Controller().type(line)
                    keyboard.Controller.press(keyboard.Controller(), key=keyboard.Key.shift_r)
                    keyboard.Controller.press(keyboard.Controller(), key=keyboard.Key.enter)
                    keyboard.Controller.release(keyboard.Controller(), key=keyboard.Key.enter)
                    keyboard.Controller.release(keyboard.Controller(), key=keyboard.Key.shift_r)
        except:    # paste what is in the clipboard anyway
            enable_application = False
            keyboard.Controller.press(keyboard.Controller(), key='v')
            keyboard.Controller.release(keyboard.Controller(), key='v')
            keyboard.Controller.release(keyboard.Controller(), key=keyboard.Key.ctrl_l)
            keyboard.Controller.release(keyboard.Controller(), key=keyboard.Key.ctrl_r)
            enable_application = True


def disable_text():     # hide / show textbox in GUI:
    global replace_new_lines_with
    if custom_replace_new_lines.get():
        textbox.config(state="normal", bg="white")
        textbox.delete('1.0', 'end')
        textbox.insert('end', replace_new_lines_with)
    else:
        replace_new_lines_with = textbox.get(1.0, "end-1c")
        textbox.delete('1.0', 'end')
        textbox.insert('end', f"{'SHIFT + RETURN': >19}")
        textbox.config(state="disabled", bg="light grey")


def set_application():
    global enable_application
    enable_application = not enable_application
    main_button["text"] = ("Disable" if enable_application else "Enable") + " application"
    main_button["bg"] = "red" if enable_application else "green"


def set_always_on_top():
    root.attributes('-topmost', always_on_top.get())


with (keyboard.Listener(on_press=on_press, on_release=on_release, win32_event_filter=win32_event_filter, suppress=False) as listener):
    root = Tk()
    root.geometry("280x150")
    root.title("Keyboard Simulator")
    font = "arial 15"

    main_button = Button(root, text=("Disable" if enable_application else "Enable") + " application", bg='red', font=font, command=set_application)
    main_button.pack()

    always_on_top = IntVar()
    always_on_top.set(True)
    Checkbutton(root, text='Always on top', font=font, variable=always_on_top, onvalue=1, offvalue=0, command=set_always_on_top).pack()
    root.attributes('-topmost', always_on_top.get())

    custom_replace_new_lines = IntVar()
    custom_replace_new_lines.set(False)
    Checkbutton(root, text='New Lines custom replace', font=font, variable=custom_replace_new_lines, onvalue=1, offvalue=0, command=disable_text).pack()

    textbox = Text(root, font=font, height=1, width=20)
    textbox.insert('end', f"{'SHIFT + RETURN': >19}")
    print()
    textbox.pack()
    textbox.config(state="disabled", bg="light grey")

    root.protocol("WM_DELETE_WINDOW", close)
    root.mainloop()
    if not stop_app:
        listener.join()
