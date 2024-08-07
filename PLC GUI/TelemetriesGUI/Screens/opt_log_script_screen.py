from kivy.core.window import Window
from kivy.graphics import Rectangle
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.screenmanager import Screen
from Screens import system_vars
from Screens.Popups.done_popup import run_done_popup
from Screens.Popups.error_popup import run_error_popup
from Screens.Popups.load_popup import load_screen
from Screens.Popups.save_popup import save_screen
from Screens.screen_manager import sm
from Scripts import opt_log_script



class OPTLogScreen(BoxLayout,Screen):
    def __init__(self, **kwargs):
        super(OPTLogScreen, self).__init__(**kwargs)
        OPTLogScreen.orientation='vertical'
        with self.canvas.before:
            self.rect=Rectangle(source="Images\LogData.JPEG",size_hint=Window.size)
        self.title = Label(text='OPT Log Analyzer', outline_color=(0, 0, 0), outline_width=5, font_size=43, font_name='Comic',size_hint_y=None, height=400)
        self.add_widget(self.title)

        #SUB GRID for save and load buttons
        self.sub_grid=BoxLayout()
        self.sub_grid.cols=2
        save_button= Button(text="Set the Save path here",size_hint_y=None, height=50,background_color=(1,1,1,0.9))
        self.sub_grid.add_widget(save_button)
        load_button =Button(text="Set the LOG file here",size_hint_y=None, height=50,background_color=(1,1,1,0.9))
        self.sub_grid.add_widget(load_button)
        self.add_widget(self.sub_grid)

        #Run and Back buttons
        run_button= Button(text="Click here to run the script",size_hint_y=None,font_size=18,outline_width=1, height=50,background_color=(0,0,1,0.8))
        self.add_widget(run_button)
        back_button=Button(text="Back to the main page",size_hint_y=None, height=50,font_size=18,outline_width=1,background_color=(1,0,0,0.8))
        self.add_widget(back_button)

        #Grid functions
        def update_rect(self,value ):
            self.rect.pos = self.pos
            self.rect.size = self.size

        def load_popup(self, *args):
            system_vars.end_file='.log'
            load_screen(*args)

        def save_popup(self, *args):
             save_screen(*args)

        def screen_transition_to_main_page(self, *args):
             system_vars.save_path=''
             system_vars.load_path=''
             sm.current = 'mainpage'

        def run_opt_script(self,*args):
            if system_vars.save_path != '':
             if system_vars.load_path != '':
              opt_log_script.runScript(system_vars.load_path,system_vars.save_path)
              run_done_popup()
            else:
                run_error_popup()


        #Binding section for event listeners
        self.bind(pos=update_rect, size=update_rect)
        load_button.bind(on_release=load_popup)
        save_button.bind(on_release=save_popup)
        back_button.bind(on_release=screen_transition_to_main_page)
        run_button.bind(on_release=run_opt_script)


