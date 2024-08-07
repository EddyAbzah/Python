import os
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.filechooser import FileChooserIconView
from kivy.uix.popup import Popup
from Screens import system_vars


def load_screen(*args):
    main_screen=BoxLayout(orientation='vertical')
    load_screen=FileChooserIconView()
    load_button=Button(size_hint_y=None,height=30)
    if system_vars.end_file=='':
        load_screen.filters=[lambda folder, filename: not filename.endswith('')]
        load_button.text="Click here to Load folder"
    elif system_vars.end_file=='.log':
        load_screen.filters = [lambda folder, filename: filename.endswith('.log')]
        load_button.text="Click here to Load file"
    main_screen.add_widget(load_screen)
    main_screen.add_widget(load_button)
    cancel_button=Button(text="Click here to go back",size_hint_y=None,height=30)
    main_screen.add_widget(cancel_button)
    load_popup_window=Popup(title='Load Menu', content=main_screen)
    load_popup_window.open()

    def loader(*args):
        if system_vars.end_file!='':
            system_vars.load_path=os.path.join(load_screen.path, load_screen.selection[0])
        else:
            system_vars.load_path=load_screen.path
        print(system_vars.load_path)
        load_popup_window.dismiss()

    def close_button(*args):
        load_popup_window.dismiss()

    cancel_button.bind(on_release=close_button)
    load_button.bind(on_release=loader)
