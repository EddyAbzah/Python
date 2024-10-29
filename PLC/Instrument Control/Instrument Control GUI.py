from kivymd.uix.button import MDFlatButton
from kivy.uix.boxlayout import BoxLayout
from kivy.lang import Builder
from kivy.core.window import Window
from kivymd.app import MDApp
from kivymd.uix.dialog import MDDialog
from kivy.properties import NumericProperty
from kivymd.uix.menu import MDDropdownMenu
import re
import SpectrumN9010B


Window.size = (1200, 800)
image_refresh_rate_ms = 1000
regex_ip_pattern = r'^(?:[0-9]{1,2}\.){3}[0-9]{1,3}$'      # pass from xx.xx.xx.xxx to x.x.x.x


class IPDialogContent(BoxLayout):
    pass


class InstrumentControlGUI(MDApp):
    spectrum = SpectrumN9010B.KeysightN9010B()      # Initialize new spectrum instance from "SpectrumN9010B.py"
    ip_address = "10.20.30.32"                      # default IP address for the Spectrum
    spectrum.use_prints = False                     # Enable terminal prints
    enable_hint_text = False                        # Enable grey text hints in the GUI's text inputs

    # Colors for the "connection status" in the top left:
    color_red = [0.7, 0.1, 0.1, 0.7]
    color_green = [0.1, 0.5, 0.1, 0.7]

    # Scrollbars and their default values:
    test_types = ["Default", "LF-PLC TX", "LF-PLC RX", "HF-PLC TX", "HF-PLC RX"]
    test_type_default = test_types[2]
    coupling_types = ["AC", "DC"]
    coupling_type_default = coupling_types[1]
    average_types = {"LogPwr": "LOG", "RMS": "RMS", "Voltage": "SCAL"}
    average_type_default = list(average_types)[1]
    trace_types = ["Average", "MaxHold", "AVG_MH"]
    trace_type_default = trace_types[2]

    # Default values:
    spectrum_range_start = 0
    spectrum_range_stop = 300
    start_frequency = NumericProperty(1.0)      # NumericProperty binds any changes to the GUI
    stop_frequency = NumericProperty(300.0)     # NumericProperty binds any changes to the GUI
    rbw = 1.0
    vbw = 1.0
    impedance = 50
    attenuation = 10
    reference_level = 10
    y_reference_level = 0

    def build(self):
        return Builder.load_file('Instrument Control GUI.kv')

    def on_start(self):
        self.show_ip_popup()  # Show IP popup at the start

        # Initialize the Test type dropdown
        self.menu_test_type = MDDropdownMenu(
            caller=self.root.ids.test_type_dropdown,
            items=[
                {"text": self.test_types[0], "viewclass": "OneLineListItem", "on_release": lambda x=0: self.set_test_type_item(0)},
                {"text": self.test_types[1], "viewclass": "OneLineListItem", "on_release": lambda x=1: self.set_test_type_item(1)},
                {"text": self.test_types[2], "viewclass": "OneLineListItem", "on_release": lambda x=2: self.set_test_type_item(2)},
                {"text": self.test_types[3], "viewclass": "OneLineListItem", "on_release": lambda x=3: self.set_test_type_item(3)},
                {"text": self.test_types[4], "viewclass": "OneLineListItem", "on_release": lambda x=4: self.set_test_type_item(4)}
            ],
            width_mult=3,
        )

        # Initialize the Coupling dropdown
        self.menu_coupling = MDDropdownMenu(
            caller=self.root.ids.coupling_dropdown,
            # Don't do this:
            # items=[{"text": v, "viewclass": "OneLineListItem", "on_release": lambda x=i: self.set_coupling_item(v)} for i, v in enumerate(self.coupling_types)],
            items=[
                {"text": self.coupling_types[0], "viewclass": "OneLineListItem", "on_release": lambda x=0: self.set_coupling_item(self.coupling_types[0])},
                {"text": self.coupling_types[1], "viewclass": "OneLineListItem", "on_release": lambda x=1: self.set_coupling_item(self.coupling_types[1])}
            ],
            width_mult=3,
        )

        # Initialize the Average type dropdown
        self.menu_avg_type = MDDropdownMenu(
            caller=self.root.ids.avg_type_dropdown,
            items=[
                {"text": list(self.average_types)[0], "viewclass": "OneLineListItem", "on_release": lambda x=0: self.set_avg_type_item(list(self.average_types.items())[0])},
                {"text": list(self.average_types)[1], "viewclass": "OneLineListItem", "on_release": lambda x=1: self.set_avg_type_item(list(self.average_types.items())[1])},
                {"text": list(self.average_types)[2], "viewclass": "OneLineListItem", "on_release": lambda x=2: self.set_avg_type_item(list(self.average_types.items())[2])}
            ],
            width_mult=3,
        )

        # Initialize the Traces dropdown
        self.menu_traces = MDDropdownMenu(
            caller=self.root.ids.traces_dropdown,
            items=[
                {"text": self.trace_types[0], "viewclass": "OneLineListItem", "on_release": lambda x=0: self.set_traces_item(self.trace_types[0])},
                {"text": self.trace_types[1], "viewclass": "OneLineListItem", "on_release": lambda x=1: self.set_traces_item(self.trace_types[1])},
                {"text": self.trace_types[2], "viewclass": "OneLineListItem", "on_release": lambda x=2: self.set_traces_item(self.trace_types[2])}
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
        pattern = re.compile(regex_ip_pattern)
        if pattern.match(ip_input) is not None:
            connection_error = self.spectrum.connect(ip_input)
            if connection_error:
                error_label.text = connection_error
            else:
                error_label.text = ""
                self.spectrum_connect(ip_input)
                self.close_dialog()
        else:
            error_label.text = "Invalid IP address. Please enter a valid one."

    def spectrum_connection(self):
        if self.root.ids.connection_button.text == "Disconnect":
            self.spectrum_disconnect()
        else:
            self.show_ip_popup()

    def spectrum_disconnect(self):
        self.spectrum.disconnect()
        self.root.ids.connection_label.text = "Disconnected"
        self.root.ids.connection_card.md_bg_color = self.color_red
        self.root.ids.connection_button.text = "Connect"

    def spectrum_connect(self, ip_input):
        self.root.ids.connection_label.text = f"Connected to {ip_input}"
        self.root.ids.connection_card.md_bg_color = self.color_green
        self.root.ids.connection_button.text = "Disconnect"

    def set_test_type_item(self, item_number):
        """Set the selected item in the Test type dropdown. Should be: "Default", "LF-PLC TX", "LF-PLC RX", "HF-PLC TX", and "HF-PLC RX"""

        # TX:
        # BW LF = 0.51
        # BW HF = 5.1

        # RX:
        # BW LF = 0.68
        # BW HF = 6.8

        self.root.ids.test_type_dropdown.text = self.test_types[item_number]
        match item_number:
            case 0:         # Default
                self.set_start_frequency(1)
                self.set_stop_frequency(10e3)
            case 1:         # LF-PLC TX
                self.set_start_frequency(50)
                self.set_stop_frequency(70)
            case 2:         # LF-PLC RX
                self.set_start_frequency(1)
                self.set_stop_frequency(300)
            case 3:         # HF-PLC TX
                self.set_start_frequency(1)
                self.set_stop_frequency(150)
            case _:         # HF-PLC RX
                self.set_start_frequency(10)
                self.set_stop_frequency(5e3)
        self.menu_test_type.dismiss()

    def set_coupling_item(self, item):
        """Set the selected item in the Coupling dropdown."""
        self.root.ids.coupling_dropdown.text = item
        self.menu_coupling.dismiss()

    def set_avg_type_item(self, item):
        """Set the selected item in the Average type dropdown."""
        item_key, item_value = item
        self.root.ids.avg_type_dropdown.text = item_key
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
        self.start_frequency = value
        # self.root.ids.start_input.text = str(value)
        if value > self.stop_frequency:
            self.update_stop_slider(value)

    def update_stop_slider(self, value):
        """Update stop slider value changes."""
        self.stop_frequency = value
        # self.root.ids.stop_input.text = str(value)
        if value < self.start_frequency:
            self.update_start_slider(value)

    def set_start_frequency(self, text):
        """Set slider start value based on input field."""
        self.start_frequency = float(text)
        # self.root.ids.slider_start.value = self.start_frequency

    def set_stop_frequency(self, text):
        """Set slider stop value based on input field."""
        self.stop_frequency = float(text)
        # self.root.ids.slider_stop.value = self.stop_frequency


if __name__ == '__main__':
    InstrumentControlGUI().run()
