# pyinstaller --noconfirm --onedir --windowed --contents-directory "Open Automation Console" --icon "transparent.png"  "Open Automation Console.py"
# This didn't work (opening the application that is a link):
# import subprocess
# application_path = r"C:\Users\eddy.a\AppData\Local\Apps\2.0\TCGE1GME.C3O\BOZV88L9.8JH\auto..tion_0000000000000000_0002.0000_ceebade57ed14022\AutomationConsole.exe"
# subprocess.Popen(application_path)

import keyboard
import pyautogui
import time

just_write = True
user = ''
key = ''


def enter_pass(package):
    package.write(user)
    package.press('tab')
    package.write(key)
    package.press('enter')


if just_write:
    # Can't use keyboard.wait() because RDP collects keyboard strokes
    time.sleep(1)
    enter_pass(keyboard)
else:
    for i in range(20):
        time.sleep(0.25)    # 20 / 0.25 = 5 second delay
        try:
            pyautogui.getWindowsWithTitle("Login")[0].activate()
            enter_pass(pyautogui)
            break
        except (Exception, ):
            pass
        try:
            pyautogui.getWindowsWithTitle("Automation Console: enter user/password")[0].activate()
            enter_pass(pyautogui)
            break
        except (Exception, ):
            pass





