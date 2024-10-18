from kivy.lang import Builder
from kivy.core.window import Window
from kivymd.app import MDApp
from kivy.properties import NumericProperty

Window.size = (800, 600)

KV = '''
BoxLayout:
    orientation: 'vertical'

    # Image Placeholder taking top half of the window
    Image:
        size_hint_y: 0.5  # Top half of the window
        source: ''  # Blank picture placeholder

    # Main Content Area
    BoxLayout:
        orientation: 'horizontal'
        size_hint_y: 0.5  # Bottom half of the window
        padding: 10

        # Left Side with Range Slider and Number Inputs
        BoxLayout:
            orientation: 'vertical'
            size_hint_x: 0.5
            spacing: 10

            # Custom Range Slider (Start and Stop Sliders) with Title
            BoxLayout:
                orientation: 'horizontal'
                size_hint_y: None
                height: '48dp'
                spacing: 10

                MDLabel:
                    text: "Range Slider: "
                    size_hint_x: None
                    width: 100

                MDSlider:
                    id: slider_start
                    min: 0
                    max: 100
                    value: app.start_value
                    step: 1
                    on_value: app.update_slider_values_from_slider('start', self.value)

                MDSlider:
                    id: slider_stop
                    min: 0
                    max: 100
                    value: app.stop_value
                    step: 1
                    on_value: app.update_slider_values_from_slider('stop', self.value)

            # Input Fields for Start and Stop Values
            BoxLayout:
                orientation: 'horizontal'
                size_hint_y: None
                height: '48dp'
                spacing: 10

                MDLabel:
                    text: "Start: "
                    size_hint_x: None
                    width: 100

                MDTextField:
                    id: start_input
                    hint_text: "Enter start value"
                    text: str(app.start_value)
                    input_filter: 'int'
                    on_text_validate: app.set_start_value(self.text)

                MDLabel:
                    text: "Stop: "
                    size_hint_x: None
                    width: 100

                MDTextField:
                    id: stop_input
                    hint_text: "Enter stop value"
                    text: str(app.stop_value)
                    input_filter: 'int'
                    on_text_validate: app.set_stop_value(self.text)

            # Number Inputs (Number 1, Number 2, and Number 3)
            BoxLayout:
                orientation: 'vertical'
                size_hint_y: None
                height: '180dp'
                spacing: 10

                BoxLayout:
                    orientation: 'horizontal'
                    size_hint_y: None
                    height: '48dp'
                    spacing: 10

                    MDLabel:
                        text: "Number 1: "
                        size_hint_x: None
                        width: 100

                    MDTextField:
                        hint_text: "Enter value"
                        input_filter: 'float'

                BoxLayout:
                    orientation: 'horizontal'
                    size_hint_y: None
                    height: '48dp'
                    spacing: 10

                    MDLabel:
                        text: "Number 2: "
                        size_hint_x: None
                        width: 100

                    MDTextField:
                        hint_text: "Enter value"
                        input_filter: 'float'

                BoxLayout:
                    orientation: 'horizontal'
                    size_hint_y: None
                    height: '48dp'
                    spacing: 10

                    MDLabel:
                        text: "Number 3: "
                        size_hint_x: None
                        width: 100

                    MDTextField:
                        hint_text: "Enter value"
                        input_filter: 'float'

        # Right Side with Dropdown Menus and Additional Inputs
        BoxLayout:
            orientation: 'vertical'
            size_hint_x: 0.5
            padding: 10
            spacing: 10

            # Dropdown Menu 1
            BoxLayout:
                orientation: 'horizontal'
                size_hint_y: None
                height: '48dp'

                MDLabel:
                    text: "Dropdown 1: "
                    size_hint_x: None
                    width: 100

                MDDropDownItem:
                    text: "Option 1"
                    on_release: self.set_item("Option 1")

            # Dropdown Menu 2
            BoxLayout:
                orientation: 'horizontal'
                size_hint_y: None
                height: '48dp'

                MDLabel:
                    text: "Dropdown 2: "
                    size_hint_x: None
                    width: 100

                MDDropDownItem:
                    text: "Option 2"
                    on_release: self.set_item("Option 2")

            # Number Inputs
            BoxLayout:
                orientation: 'horizontal'
                size_hint_y: None
                height: '48dp'

                MDLabel:
                    text: "Number 4: "
                    size_hint_x: None
                    width: 100

                MDTextField:
                    hint_text: "Enter value"
                    input_filter: 'float'

            BoxLayout:
                orientation: 'horizontal'
                size_hint_y: None
                height: '48dp'

                MDLabel:
                    text: "Number 5: "
                    size_hint_x: None
                    width: 100

                MDTextField:
                    hint_text: "Enter value"
                    input_filter: 'float'
'''

class MyApp(MDApp):
    start_value = NumericProperty(20)
    stop_value = NumericProperty(80)

    def build(self):
        return Builder.load_string(KV)

    def update_slider_values_from_slider(self, slider_type, value):
        """Update input fields based on slider value."""
        if slider_type == 'start':
            self.start_value = value
            self.root.ids.start_input.text = str(value)
        elif slider_type == 'stop':
            self.stop_value = value
            self.root.ids.stop_input.text = str(value)

    def set_start_value(self, value):
        """Set slider value based on input field."""
        try:
            value = int(value)
            if 0 <= value <= self.stop_value:
                self.start_value = value
                self.root.ids.slider_start.value = value
        except ValueError:
            pass

    def set_stop_value(self, value):
        """Set slider value based on input field."""
        try:
            value = int(value)
            if value >= self.start_value and value <= 100:
                self.stop_value = value
                self.root.ids.slider_stop.value = value
        except ValueError:
            pass

if __name__ == '__main__':
    MyApp().run()
