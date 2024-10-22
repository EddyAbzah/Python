from kivy.lang import Builder
from kivy.core.window import Window
from kivymd.app import MDApp
from kivy.properties import NumericProperty
from kivymd.uix.menu import MDDropdownMenu

Window.size = (1200, 800)
image_refresh_rate_ms = 1000


class InstrumentControlGUI(MDApp):
    start_value = NumericProperty(20)
    stop_value = NumericProperty(80)

    def build(self):
        return Builder.load_file('Instrument Control GUI.kv')

    def on_start(self):
        # Initialize the Test type dropdown
        self.menu_test_type = MDDropdownMenu(
            caller=self.root.ids.test_type_dropdown,
            items=[
                {"text": "Full range", "viewclass": "OneLineListItem", "on_release": lambda x="Full range": self.set_test_type_item("Full range")},
                {"text": "LF-PLC", "viewclass": "OneLineListItem", "on_release": lambda x="LF-PLC": self.set_test_type_item("LF-PLC")},
                {"text": "HF-PLC", "viewclass": "OneLineListItem", "on_release": lambda x="HF-PLC": self.set_test_type_item("HF-PLC")}
            ],
            width_mult=3,
        )

        # Initialize the Coupling dropdown
        self.menu_coupling = MDDropdownMenu(
            caller=self.root.ids.coupling_dropdown,
            items=[
                {"text": "AC", "viewclass": "OneLineListItem", "on_release": lambda x="AC": self.set_coupling_item("AC")},
                {"text": "DC", "viewclass": "OneLineListItem", "on_release": lambda x="DC": self.set_coupling_item("DC")}
            ],
            width_mult=3,
        )

        # Initialize the Average type dropdown
        self.menu_avg_type = MDDropdownMenu(
            caller=self.root.ids.avg_type_dropdown,
            items=[
                {"text": "Log", "viewclass": "OneLineListItem", "on_release": lambda x="Log": self.set_avg_type_item("Log")},
                {"text": "Pwr", "viewclass": "OneLineListItem", "on_release": lambda x="Pwr": self.set_avg_type_item("Pwr")},
                {"text": "Temp", "viewclass": "OneLineListItem", "on_release": lambda x="Temp": self.set_avg_type_item("Temp")}
            ],
            width_mult=3,
        )

        # Initialize the Traces dropdown
        self.menu_traces = MDDropdownMenu(
            caller=self.root.ids.traces_dropdown,
            items=[
                {"text": "Average", "viewclass": "OneLineListItem", "on_release": lambda x="Average": self.set_traces_item("Average")},
                {"text": "MaxHold", "viewclass": "OneLineListItem", "on_release": lambda x="MaxHold": self.set_traces_item("MaxHold")},
                {"text": "AVG_MH", "viewclass": "OneLineListItem", "on_release": lambda x="AVG_MH": self.set_traces_item("AVG_MH")}
            ],
            width_mult=3,
        )

    def set_test_type_item(self, item):
        """Set the selected item in the Test type dropdown."""
        self.root.ids.test_type_dropdown.text = item
        self.menu_test_type.dismiss()

    def set_coupling_item(self, item):
        """Set the selected item in the Coupling dropdown."""
        self.root.ids.coupling_dropdown.text = item
        self.menu_coupling.dismiss()

    def set_avg_type_item(self, item):
        """Set the selected item in the Average type dropdown."""
        self.root.ids.avg_type_dropdown.text = item
        self.menu_avg_type.dismiss()

    def set_traces_item(self, item):
        """Set the selected item in the Traces dropdown."""
        self.root.ids.traces_dropdown.text = item
        self.menu_traces.dismiss()

    def config_spectrum(self):
        """Placeholder for config spectrum method."""
        print("Config spectrum method called.")

    def reset_spectrum(self):
        """Placeholder for reset spectrum method."""
        print("Reset spectrum method called.")

    def run_traces(self):
        """Placeholder for run traces method."""
        print("Run traces method called.")

    def stop_traces(self):
        """Placeholder for stop traces method."""
        print("Stop traces method called.")

    def update_slider_values_from_slider(self, slider_type, value):
        """Update slider value changes."""
        if slider_type == 'start':
            self.start_value = value
            self.root.ids.start_input.text = str(value)
        elif slider_type == 'stop':
            self.stop_value = value
            self.root.ids.stop_input.text = str(value)

    def set_start_value(self, text):
        """Set slider start value based on input field."""
        self.start_value = int(text)
        self.root.ids.slider_start.value = self.start_value

    def set_stop_value(self, text):
        """Set slider stop value based on input field."""
        self.stop_value = int(text)
        self.root.ids.slider_stop.value = self.stop_value


if __name__ == '__main__':
    InstrumentControlGUI().run()
