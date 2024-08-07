from kivy.uix.video import Video
from kivy.core.window import Window
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.screenmanager import Screen
from Screens.screen_manager import sm


class VideoLayout(BoxLayout, Screen):
    def __init__(self, **kwargs):
        super(VideoLayout, self).__init__(**kwargs)
        VideoLayout.orientation = 'horizontal'
        with self.canvas.before:
            video = Video(source=".\Videos\DBZ.mp4")
            video.state = 'play'
            video.size = Window.size

        # start_button = Button(text='Click here to start',size_hint_y=None, height=50,background_color=(0,0,1,0.5))
        # self.add_widget(start_button)

        # Grid functions
        def on_position_change(self, value):
            if (duration - 0.1 < value):
                video.state = 'stop'
                sm.current = 'mainpage'

        def on_duration_change(self, value):
            global duration
            duration = value

        def update_rect(self, value):
            video.size = Window.size

        def screen_transition_load(self, *args):
            video.state = 'stop'
            sm.current = 'mainpage'

        # Binding section for event listeners
        video.bind(position=on_position_change, duration=on_duration_change)
        # start_button.bind(on_press=screen_transition_load)
        self.bind(pos=update_rect, size=update_rect)

