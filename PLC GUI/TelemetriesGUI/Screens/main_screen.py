from kivy.core.window import Window
from kivy.graphics import Rectangle
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.screenmanager import Screen
from Screens.screen_manager import sm


class MainScreen(BoxLayout,Screen):
    def __init__(self, **kwargs):
        super(MainScreen, self).__init__(**kwargs)
        MainScreen.orientation='vertical'
        with self.canvas.before:
            self.rect=Rectangle(source="Images\MainPageLogo.JPG",size_hint=Window.size)
        self.title = Label(text='Welcome to the main page!', outline_color=(0, 0, 0), outline_width=4, font_size=36, font_name='Comic',size_hint_y=None, height=-450)
        self.add_widget(self.title)
        self.title2 = Label(text='Choose your App', outline_color=(0, 0, 0), outline_width=4, font_size=24, font_name='Comic',size_hint_y=None, height=550)
        self.add_widget(self.title2)

        #Buttons for each App
        choose_alon_hason = Button(text='Alon Hason APP',size_hint_y=None, height=50,background_color=(1,1,1,0.8))
        self.add_widget(choose_alon_hason)
        choose_telemetries = Button(text='Telemetries Log Analyzer',size_hint_y=None, height=50,background_color=(1,1,1,0.8))
        self.add_widget(choose_telemetries)
        choose_opt_log_script = Button(text='OPT Log Analyzer', size_hint_y=None, height=50,background_color=(1,1,1,0.8))
        self.add_widget(choose_opt_log_script)

        #Grid functions
        def screen_transition_to_opt_app(self, *args):
             sm.current = 'optlog'

        def screen_transition_to_telemetries_app(self, *args):
             sm.current = 'telemetries'

        def screen_transition_to_alon_hason_app(self, *args):
             sm.current = 'alon_hason'

        def update_rect(self,value ):
            self.rect.pos = self.pos
            self.rect.size = self.size

        #Binding section for event listeners
        choose_opt_log_script.bind(on_release=screen_transition_to_opt_app)
        choose_telemetries.bind(on_release=screen_transition_to_telemetries_app)
        choose_alon_hason.bind(on_release=screen_transition_to_alon_hason_app)
        self.bind(pos=update_rect, size=update_rect)
