from kivy.core.window import Window
from kivy.graphics import Rectangle
from kivy.uix.button import Button
from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label
from kivy.uix.screenmanager import Screen
from kivy.uix.textinput import TextInput
from Screens import system_vars
from Screens.Popups.error_popup import run_error_popup
from Screens.Popups.load_popup import load_screen
from Screens.Popups.save_popup import save_screen
from Screens.Popups.wait_while_processing_popup import run_wait_popup
from Screens.screen_manager import sm
from Scripts import telemetries_log_script


class TelemtriesWindow(GridLayout,Screen):
    def __init__(self, **kwargs):
        super(TelemtriesWindow, self).__init__(**kwargs)
        with self.canvas:
            self.rect=Rectangle(source="Images\TelemetriesLogo.JPG",size_hint=Window.size, keep_ratio=True)
        self.rows=6
        self.cols=1
        self.title=Label(text='Telemetries Log Analyzer',outline_color=(0,0,0),outline_width=4 ,  font_size=36,font_name='Comic')
        self.add_widget(self.title)


       #Text input for OPT ID
        self.instructions=Label(text="Enter The OPT list separated by Comma:", outline_color=(0,0,0),outline_width=4,font_size=25,size_hint_y=None,height=40)
        self.add_widget(self.instructions)
        opt_input=TextInput(size_hint_y=None,height=40,multiline=False)
        self.add_widget(opt_input)

        #SUB GRID for save and load buttons
        self.second_grid=GridLayout()
        self.second_grid.size_hint_y=None
        self.second_grid.height=40
        self.second_grid.cols = 2
        save_button= Button(text="Set the Save path here",outline_color=(0,0,0),outline_width=1,size_hint_y=None,height=40,font_size=18,background_color=(1,1,1,0.85))
        self.second_grid.add_widget(save_button)
        load_button= Button(text="Set the Excel files folder here",outline_color=(0,0,0),outline_width=1,size_hint_y=None,height=40,font_size=18,background_color=(1,1,1,0.85))
        self.second_grid.add_widget(load_button)
        self.add_widget(self.second_grid)

       #Run and Back buttons
        run_button= Button(text="Click here to run the script",outline_color=(0,0,0),outline_width=1,size_hint_y=None, height=50, font_size=18,background_color=(0,0,1,0.8))
        self.add_widget(run_button)
        back_button=Button(text="Back to the main page",outline_color=(0,0,0),outline_width=1,size_hint_y=None, height=50, font_size=18,background_color=(1,0,0,0.8))
        self.add_widget(back_button)

        #Grid functions
        def update_rect(self,value ):
            self.rect.pos = self.pos
            self.rect.size = self.size

        def screen_transition_to_main_page(self, *args):
             system_vars.save_path=''
             system_vars.load_path=''
             sm.current = 'mainpage'

        def load_popup(self, *args):
            system_vars.end_file=''
            load_screen()

        def save_popup(self, *args):
             save_screen(*args)

        def run_excel_script(self,*args):
            if system_vars.save_path !='':
                if system_vars.load_path!='':
                    run_wait_popup()
                    telemetries_log_script.excel_to_df(system_vars.load_path,system_vars.save_path,opt_input.text)
            else:
                run_error_popup()

        #Binding section for event listeners
        load_button.bind(on_release=load_popup)
        save_button.bind(on_release=save_popup)
        run_button.bind(on_release=run_excel_script)
        back_button.bind(on_release=screen_transition_to_main_page)
        self.bind(pos=update_rect, size=update_rect)

