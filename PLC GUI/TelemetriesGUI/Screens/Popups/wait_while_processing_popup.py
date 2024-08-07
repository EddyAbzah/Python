from tkinter import Label
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.popup import Popup


def run_wait_popup():
    content=BoxLayout(orientation='vertical')
    text=Label(text="Please wait while processing")
    content.add_widget(text)
    popupWindow = Popup(title="Working!", content=content, size_hint=(None,None),size=(400,200))
    popupWindow.open()