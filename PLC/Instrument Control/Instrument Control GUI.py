from kivymd.uix.button import MDFlatButton
from kivy.uix.boxlayout import BoxLayout
from kivy.lang import Builder
from kivy.core.window import Window
from kivymd.app import MDApp
from kivymd.uix.dialog import MDDialog
from kivy.properties import NumericProperty
from kivymd.uix.menu import MDDropdownMenu
import re

Window.size = (1200, 800)
image_refresh_rate_ms = 1000
test_types = ["Full range", "LF-PLC", "HF-PLC"]


class IPDialogContent(BoxLayout):
    pass


class InstrumentControlGUI(MDApp):
    start_value = NumericProperty(20)
    stop_value = NumericProperty(80)
    color_red = [0.7, 0.1, 0.1, 0.7]
    color_green = [0.1, 0.5, 0.1, 0.7]

    def build(self):
        return Builder.load_file('Instrument Control GUI.kv')

    def on_start(self):
        self.show_ip_popup()  # Show IP popup at the start

        # Initialize the Test type dropdown
        self.menu_test_type = MDDropdownMenu(
            caller=self.root.ids.test_type_dropdown,
            items=[
                {"text": test_types[0], "viewclass": "OneLineListItem", "on_release": lambda x="Full range": self.set_test_type_item("Full range")},
                {"text": test_types[1], "viewclass": "OneLineListItem", "on_release": lambda x="LF-PLC": self.set_test_type_item("LF-PLC")},
                {"text": test_types[2], "viewclass": "OneLineListItem", "on_release": lambda x="HF-PLC": self.set_test_type_item("HF-PLC")}
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

    def show_ip_popup(self):
        """Show a popup for entering the IP address."""
        self.dialog = MDDialog(
            title="Enter IP",
            type="custom",
            content_cls=IPDialogContent(),
            buttons=[
                MDFlatButton(text="Cancel", on_release=self.close_dialog),
                MDFlatButton(text="OK", on_release=self.check_ip_validity),
            ],
        )
        self.dialog.open()

    def close_dialog(self, *_):
        """Close the popup dialog."""
        self.dialog.dismiss()

    def check_ip_validity(self, *_):
        """Check if the entered IP address is valid."""
        error_label = self.dialog.content_cls.ids.error_label
        ip_input = self.dialog.content_cls.ids.ip_input.text
        pattern = re.compile(r'^[0-9]{2}\.[0-9]\.[0-9]\.[0-9]{1,3}$')      # = xx.x.x.x or xx.x.x.xx    original = (r'^(?:[0-9]{2}\.){3}[0-9]{1,2}$')
        if pattern.match(ip_input) is not None:
            error_label.text = ""
            self.spectrum_connect(ip_input)
            self.close_dialog()
        else:
            error_label.text = "Invalid IP address. Please enter a valid one."
            print("Invalid IP address. Please enter a valid one.")

    def spectrum_connection(self):
        if self.root.ids.connection_button.text == "Disconnect":
            self.spectrum_disconnect()
        else:
            self.show_ip_popup()

    def spectrum_disconnect(self):
        self.root.ids.connection_label.text = "Disconnected"
        self.root.ids.connection_card.md_bg_color = self.color_red
        self.root.ids.connection_button.text = "Connect"

    def spectrum_connect(self, ip_input):
        self.root.ids.connection_label.text = f"Connected to {ip_input}"
        self.root.ids.connection_card.md_bg_color = self.color_green
        self.root.ids.connection_button.text = "Disconnect"

    def set_test_type_item(self, item):
        """Set the selected item in the Test type dropdown."""
        self.root.ids.test_type_dropdown.text = item
        if item == test_types[1]:
            self.set_start_value(10)
            self.set_stop_value(90)
        elif item == test_types[2]:
            self.set_start_value(20)
            self.set_stop_value(80)
        else:
            self.set_start_value(0)
            self.set_stop_value(100)
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

    def update_start_slider(self, value):
        """Update start slider value changes."""
        self.start_value = value
        self.root.ids.start_input.text = str(value)
        if value > self.stop_value:
            self.update_stop_slider(value)

    def update_stop_slider(self, value):
        """Update stop slider value changes."""
        self.stop_value = value
        self.root.ids.stop_input.text = str(value)
        if value < self.start_value:
            self.update_start_slider(value)

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
