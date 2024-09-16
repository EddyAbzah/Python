from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.popup import Popup
import playsound


def run_error_popup():
    content=BoxLayout(orientation='vertical')
    text=Label(text="You forgot to define save or load path, please try again")
    content.add_widget(text)
    close_button=Button(text="Close",size_hint_y=None ,height=50)
    content.add_widget(close_button)
    popupWindow = Popup(title="Error!", content=content, size_hint=(None,None),size=(400,200))
    playsound.playsound("Sound\LO.mp3")
    popupWindow.open()


    def close_pop(*args):
        popupWindow.dismiss()

    close_button.bind(on_release=close_pop)
