from kivy.lang import Builder
from kivy.core.window import Window
from kivymd.app import MDApp
from kivy.properties import NumericProperty

Window.size = (800, 600)

KV = '''
BoxLayout:
    orientation: 'vertical'

    # Image taking top half of the window
    Image:
        size_hint_y: 0.5  # Top half of the window
        source: 'Temp pic.png'  # Use your image here

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
                    text: "Frequency Range [kHz]: "
                    size_hint_x: None
                    width: 150

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
                    text: "Start Frequency [kHz]: "
                    size_hint_x: None
                    width: 150

                MDTextField:
                    id: start_input
                    hint_text: "Enter start frequency"
                    text: str(app.start_value)
                    input_filter: 'int'
                    on_text_validate: app.set_start_value(self.text)

                MDLabel:
                    text: "Stop Frequency [kHz]: "
                    size_hint_x: None
                    width: 150

                MDTextField:
                    id: stop_input
                    hint_text: "Enter stop frequency"
                    text: str(app.stop_value)
                    input_filter: 'int'
                    on_text_validate: app.set_stop_value(self.text)

            # Number Inputs (RBW, VBW, Test type, and Impedance)
            BoxLayout:
                orientation: 'vertical'
                size_hint_y: None
                height: '220dp'
                spacing: 10

                BoxLayout:
                    orientation: 'horizontal'
                    size_hint_y: None
                    height: '48dp'
                    spacing: 10

                    MDLabel:
                        text: "RBW [kHz]: "
                        size_hint_x: None
                        width: 150

                    MDTextField:
                        hint_text: "Enter RBW"
                        input_filter: 'float'

                BoxLayout:
                    orientation: 'horizontal'
                    size_hint_y: None
                    height: '48dp'
                    spacing: 10

                    MDLabel:
                        text: "VBW [kHz]: "
                        size_hint_x: None
                        width: 150

                    MDTextField:
                        hint_text: "Enter VBW"
                        input_filter: 'float'

                # Test type Dropdown Menu
                BoxLayout:
                    orientation: 'horizontal'
                    size_hint_y: None
                    height: '48dp'

                    MDLabel:
                        text: "Test type: "
                        size_hint_x: None
                        width: 150

                    MDDropDownItem:
                        text: "Select Test type"
                        on_release: self.set_item("Select Test type")

                BoxLayout:
                    orientation: 'horizontal'
                    size_hint_y: None
                    height: '48dp'
                    spacing: 10

                    MDLabel:
                        text: "Impedance [ohm]: "
                        size_hint_x: None
                        width: 150

                    MDTextField:
                        hint_text: "Enter Impedance"
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
                    text: "Coupling: "
                    size_hint_x: None
                    width: 150

                MDDropDownItem:
                    text: "Select Coupling"
                    on_release: self.set_item("Select Coupling")

            # Dropdown Menu 2
            BoxLayout:
                orientation: 'horizontal'
                size_hint_y: None
                height: '48dp'

                MDLabel:
                    text: "Average type: "
                    size_hint_x: None
                    width: 150

                MDDropDownItem:
                    text: "Select Average type"
                    on_release: self.set_item("Select Average type")

            # Number Inputs
            BoxLayout:
                orientation: 'horizontal'
                size_hint_y: None
                height: '48dp'

                MDLabel:
                    text: "Attenuation [dB]: "
                    size_hint_x: None
                    width: 150

                MDTextField:
                    hint_text: "Enter Attenuation"
                    input_filter: 'float'

            BoxLayout:
                orientation: 'horizontal'
                size_hint_y: None
                height: '48dp'

                MDLabel:
                    text: "Reference Level [dBm]: "
                    size_hint_x: None
                    width: 150

                MDTextField:
                    hint_text: "Enter Reference Level"
                    input_filter: 'float'

            BoxLayout:
                orientation: 'horizontal'
                size_hint_y: None
                height: '48dp'

                MDLabel:
                    text: "Y Reference Level [dBm]: "
                    size_hint_x: None
                    width: 150

                MDTextField:
                    hint_text: "Enter Y Reference Level"
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
