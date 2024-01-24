# pyinstaller --onefile --noconsole --icon Salvador.ico --add-data "C:/Users/eddy.a/OneDrive - SolarEdge/Documents/Python Scripts/Sahar/TelemetriesGUI 03/Media;Media/"  Salvador.py


import os
import sys
import playsound
from kivy.app import App
from kivy.config import Config
from kivy.core.window import Window
from kivy.graphics import Rectangle
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.gridlayout import GridLayout
from kivy.uix.screenmanager import Screen


def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    relative_path = "Media\\" + relative_path
    print(f'{os.path.join(base_path, relative_path)}')
    return os.path.join(base_path, relative_path)


class AlonHason(BoxLayout, Screen):
    def __init__(self, **kwargs):
        super(AlonHason, self).__init__(**kwargs)
        AlonHason.orientation = 'vertical'
        with self.canvas.before:
            self.rect = Rectangle(source=resource_path("Salvador.JPG"), size_hint=Window.size)

        self.sub_grid = GridLayout()
        self.sub_grid.cols = 2
        height = 150
        background_color = (0, 0, 0, 0)

        kama_button = Button(size_hint_y=None, height=height, background_color=background_color)
        self.sub_grid.add_widget(kama_button)

        tagidi_kama_button = Button(size_hint_y=None, height=height, background_color=background_color)
        self.sub_grid.add_widget(tagidi_kama_button)

        lama_button = Button(size_hint_y=None, height=height, background_color=background_color)
        self.sub_grid.add_widget(lama_button)

        az_button = Button(size_hint_y=None, height=height, background_color=background_color)
        self.sub_grid.add_widget(az_button)

        salvador_button = Button(size_hint_y=None, height=height, background_color=background_color)
        self.sub_grid.add_widget(salvador_button)

        moser_button = Button(size_hint_y=None, height=height, background_color=background_color)
        self.sub_grid.add_widget(moser_button)

        adhaerev_button = Button(size_hint_y=None, height=height, background_color=background_color)
        self.sub_grid.add_widget(adhaerev_button)

        byebye_button = Button(size_hint_y=None, height=height, background_color=background_color)
        self.sub_grid.add_widget(byebye_button)

        self.add_widget(self.sub_grid)

        def byebye_function(instance):
            sys.exit()

        def tagidi_kama_function(instance):
            playsound.playsound(resource_path("Shalom Leha Habibi.mp3"))

        def lama_function(instance):
            playsound.playsound(resource_path("Medaber Salvador.mp3"))

        def kama_function(instance):
            playsound.playsound(resource_path("Check Katan.mp3"))

        def az_function(instance):
            playsound.playsound(resource_path("AZ Ani modia Lehane motek.mp3"))

        def moser_function(instance):
            playsound.playsound(resource_path("Moser Letipul.mp3"))

        def adhaerev_function(instance):
            playsound.playsound(resource_path("Ad HaHerev.mp3"))

        def salvador_function(instance):
            playsound.playsound(resource_path("Salvadorr.mp3"))

        def update_rect(instance, value):
            self.rect.pos = self.pos
            self.rect.size = self.size

        # Binding section for event listeners:
        kama_button.bind(on_press=kama_function)
        lama_button.bind(on_press=lama_function)
        az_button.bind(on_press=az_function)
        salvador_button.bind(on_press=salvador_function)
        moser_button.bind(on_press=moser_function)
        adhaerev_button.bind(on_press=adhaerev_function)
        byebye_button.bind(on_press=byebye_function)
        tagidi_kama_button.bind(on_press=tagidi_kama_function)
        self.bind(pos=update_rect, size=update_rect)


Config.set('graphics', 'resizable', '0')
Window.borderless = True


class MyApp(App):
    def build(self):
        return AlonHason(name='alon_hason')


if __name__ == "__main__":
    MyApp().run()
