# pyinstaller --noconfirm --onedir --windowed --contents-directory "Monitors Off" --icon "transparent.png"  "Monitors Off.py"

import ctypes
import time


def turn_off_monitor():
    ctypes.windll.user32.SendMessageW(0xFFFF, 0x0112, 0xF170, 2)


time.sleep(1)
turn_off_monitor()
