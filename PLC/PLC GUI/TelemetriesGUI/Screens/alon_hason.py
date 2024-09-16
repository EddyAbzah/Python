from kivy.core.window import Window
from kivy.graphics import Rectangle
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.gridlayout import GridLayout
from kivy.uix.screenmanager import Screen
from Screens.screen_manager import sm
import playsound


class AlonHason(BoxLayout,Screen):
    def __init__(self, **kwargs):
        super(AlonHason, self).__init__(**kwargs)
        AlonHason.orientation='vertical'
        with self.canvas.before:
            self.rect=Rectangle(source="Images\Salvador.JPG",size_hint=Window.size)
        #Buttons for each App

        self.sub_grid=GridLayout()
        self.sub_grid.cols=4
        kama_button = Button(text='Check Katan',size_hint_y=None, height=50,background_color=(1,1,1,0.5))
        self.sub_grid.add_widget(kama_button)
        tagidi_kama_button = Button(text='Shalom_leha_habibi',size_hint_y=None, height=50,background_color=(1,1,1,0.5))
        self.sub_grid.add_widget(tagidi_kama_button)
        lama_button = Button(text='Medaber Salvador',size_hint_y=None, height=50,background_color=(1,1,1,0.5))
        self.sub_grid.add_widget(lama_button)
        az_button = Button(text='Az Ani Modia Leha Motek',size_hint_y=None, height=50,background_color=(1,1,1,0.5))
        self.sub_grid.add_widget(az_button)
        salvador_button = Button(text='Salvadorr',size_hint_y=None, height=50,background_color=(1,1,1,0.5))
        self.sub_grid.add_widget(salvador_button)
        moser_button = Button(text='Moser Letipul',size_hint_y=None, height=50,background_color=(1,1,1,0.5))
        self.sub_grid.add_widget(moser_button)
        adhaerev_button = Button(text='Ad HaHerev',size_hint_y=None, height=50,background_color=(1,1,1,0.5))
        self.sub_grid.add_widget(adhaerev_button)
        self.add_widget(self.sub_grid)

        go_back_button = Button(text='Go Back', size_hint_y=None, height=50)
        self.sub_grid.add_widget(go_back_button)


        #Grid functions
        def tagidi_kama_function(self, *args):
            playsound.playsound("Sound\Shalom Leha Habibi.mp3")


        def lama_function(self, *args):
            playsound.playsound("Sound\Medaber Salvador.mp3")

        def kama_function(self, *args):
            playsound.playsound("Sound\Check Katan.mp3")

        def az_function(self, *args):
            playsound.playsound("Sound\AZ Ani modia Lehane motek.mp3")

        def moser_function(self, *args):
            playsound.playsound("Sound\Moser Letipul.mp3")

        def adhaerev_function(self, *args):
            playsound.playsound("Sound\Ad HaHerev.mp3")

        def salvador_function(self, *args):
            playsound.playsound("Sound\Salvadorr.mp3")

        def go_back(self, *args):
             sm.current = 'mainpage'

        def update_rect(self,value ):
            self.rect.pos = self.pos
            self.rect.size = self.size

        #Binding section for event listeners
        kama_button.bind(on_press=kama_function)
        lama_button.bind(on_press=lama_function)
        az_button.bind(on_press=az_function)
        salvador_button.bind(on_press=salvador_function)
        moser_button.bind(on_press=moser_function)
        adhaerev_button.bind(on_press=adhaerev_function)
        tagidi_kama_button.bind(on_press=tagidi_kama_function)
        go_back_button.bind(on_release=go_back)
        self.bind(pos=update_rect, size=update_rect)
