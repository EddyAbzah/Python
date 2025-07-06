# pyinstaller --noconfirm --onedir --windowed --contents-directory "Toggle HDR" --icon "transparent.png"  "Toggle HDR.py"

import pyautogui

pyautogui.keyDown('winleft')
pyautogui.keyDown('alt')
pyautogui.press('b')
pyautogui.keyUp('alt')
pyautogui.keyUp('winleft')
