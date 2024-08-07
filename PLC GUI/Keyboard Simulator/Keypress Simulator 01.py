# pyinstaller --onefile --noconsole --icon "icon.ico" --add-data "Hagiga in the Snooker.mp3;."  "Keypress Simulator 02.py"


import os
import sys
import time
from pygame import mixer
from tkinter import Tk, Button, Canvas
from pynput import keyboard

stop_app = False


def resource_path(relative_path):
    base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)


def close():
    global stop_app
    stop_app = True
    listener.stop()
    root.destroy()
    sys.exit()


def on_press(key):
    if key == keyboard.Key.esc:
        close()
    elif type(key) is type(keyboard.KeyCode()) and '\\' in repr(key) and ord(key.char) == 22:
        text = Tk().clipboard_get()
        time.sleep(1)
        keyboard.Controller().type(text)


def play_music():
    if button["text"] == "Play":
        mixer.music.unpause()
        button["text"] = "Pause"
        button["bg"] = "red"
    else:
        mixer.music.pause()
        button["text"] = "Play"
        button["bg"] = "green"


with keyboard.Listener(on_press=on_press) as listener:
    mixer.init()
    mixer.music.load(resource_path("Hagiga in the Snooker.mp3"))
    mixer.music.play()
    root = Tk()
    root.geometry("500x300")
    root.title("Keyboard Simulator")
    Canvas = Canvas(root, width=500, height=150)
    Canvas.pack()
    Canvas.create_text(250, 75, font="cmr 18 bold", fill='black', text="    The music is only to remind you\nto close the app after you've finished")
    button = Button(root, text='Pause', bg='red', font="cmr 20", command=play_music)
    button.pack()
    root.protocol("WM_DELETE_WINDOW", close)
    root.mainloop()
    if not stop_app:
        listener.join()
