from kivy.config import Config
Config.set('graphics', 'resizable', '0')

from kivy.app import App
from Screens.main_screen import MainScreen
from Screens.opt_log_script_screen import OPTLogScreen
from Screens.telemtries_log_script_screen import TelemtriesWindow
from Screens.video_screen import VideoLayout
from Screens.alon_hason import AlonHason
from Screens.screen_manager import sm


class MyApp(App):
    def build(self):
        return sm


if __name__ == "__main__":
    sm.add_widget(AlonHason(name='alon_hason'))
    sm.add_widget(MainScreen(name='mainpage'))
    sm.add_widget(OPTLogScreen(name='optlog'))
    sm.add_widget(TelemtriesWindow(name='telemetries'))
    MyApp().run()