import os
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.filechooser import FileChooserIconView
from kivy.uix.label import Label
from kivy.uix.popup import Popup
from kivy.uix.textinput import TextInput
from Screens import system_vars



def save_screen(*args):
    main_screen=BoxLayout(orientation='vertical')
    save_screen=FileChooserIconView(filters=[lambda folder, filename: not filename.endswith('')])
    main_screen.add_widget(save_screen)
    instructions=Label(text="Enter your file name",size_hint_y=None,height=30)
    main_screen.add_widget(instructions)
    file_name=TextInput(size_hint_y=None,height=30,multiline=False)
    main_screen.add_widget(file_name)
    save_button=Button(text="Click here to Save",size_hint_y=None,height=30)
    main_screen.add_widget(save_button)
    cancel_button=Button(text="Click here to go back",size_hint_y=None,height=30)
    main_screen.add_widget(cancel_button)
    save_popup_window=Popup(title='Save Menu', content=main_screen)
    save_popup_window.open()

    def saver(*args):
        system_vars.save_path=os.path.join(save_screen.path,file_name.text)
        system_vars.dir=save_screen.path
        save_popup_window.dismiss()

    def close_button(*args):
        save_popup_window.dismiss()

    cancel_button.bind(on_release=close_button)
    save_button.bind(on_release=saver)