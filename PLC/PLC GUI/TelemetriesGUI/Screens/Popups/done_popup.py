from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.popup import Popup
from Screens import system_vars


def run_done_popup():
    content=BoxLayout(orientation='vertical')
    text=Label(text="The file is ready at "+system_vars.dir)
    content.add_widget(text)
    close_button=Button(text="Close",size_hint_y=None ,height=50)
    content.add_widget(close_button)
    popupWindow = Popup(title="Succeeded!", content=content, size_hint=(None,None),size=(400,200))
    popupWindow.open()

    def close_pop(*args):
        popupWindow.dismiss()

    close_button.bind(on_release=close_pop)