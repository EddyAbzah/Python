import re
from tabulate import tabulate
from kivy.utils import platform
from kivymd.uix.button import MDFlatButton
from kivy.uix.boxlayout import BoxLayout
from kivy.lang import Builder
from kivy.core.window import Window
from kivymd.app import MDApp
from kivymd.uix.dialog import MDDialog
from kivy.properties import NumericProperty, BooleanProperty
from kivymd.uix.menu import MDDropdownMenu
import Spectrum_Keysight_N9010


set_window = [True,                 # to set, or not to set
              1,                    # Refactor screen
              [[1080, 2176, 409],   # 0 = Xiaomi Redmi Note 11S - Portrait
               [2176, 986, 409],    # 1 = Xiaomi Redmi Note 11S - Landscape
               [1080, 2268, 398],   # 2 = Xiaomi Mi Note 10 Pro - Portrait
               [2268, 1080, 398],   # 3 = Xiaomi Mi Note 10 Pro - Landscape
               [1008, 2076, 489],   # 4 = Google Pixel 8 Pro - Portrait
               [2130, 890, 489],    # 5 = Google Pixel 8 Pro - Landscape
               [600, 800, 200],     # 6 = Custom 01
               [1200, 800, 100]]    # 7 = PC
              [7]]                  # pick from the phones above


if platform == "android":
    from jnius import autoclass
    PythonActivity = autoclass("org.kivy.android.PythonActivity")
    ActivityInfo = autoclass("android.content.pm.ActivityInfo")
    activity = PythonActivity.mActivity
    activity.setRequestedOrientation(ActivityInfo.SCREEN_ORIENTATION_USER)
elif set_window[0]:
    Window.size = int(set_window[2][0] / set_window[1]), int(set_window[2][1] / set_window[1])
    Window.dpi = int(set_window[2][2] / set_window[1])


image_refresh_rate_ms = 1000
regex_ip_pattern = r'^(?:[0-9]{1,2}\.){3}[0-9]{1,3}$'      # pass from xx.xx.xx.xxx to x.x.x.x


class IPDialogContent(BoxLayout):
    pass


class MismatchDialogContent(BoxLayout):
    pass


class InstrumentControlGUI(MDApp):
    spectrum = Spectrum_Keysight_N9010.KeysightN9010B()         # Initialize new spectrum instance from "SpectrumN9010B.py"
    is_connected = False                                        # True if the Spectrum is connected
    ip_address = "10.20.30.49"                                  # default IP address for the Spectrum
    spectrum.use_prints = True                                  # Enable terminal prints
    enable_hint_text = True                                     # Enable grey text hints in the GUI's text inputs
    is_scrollable = BooleanProperty(False)                      # if is_scrollable = True, the lables will be split to two columns

    # Colors for the "connection status" in the top left:
    color_red = [0.7, 0.1, 0.1, 0.7]
    color_green = [0.1, 0.5, 0.1, 0.7]

    # Scrollbars values:
    test_types = ["Default", "LF-PLC TX", "LF-PLC RX", "HF-PLC TX", "HF-PLC RX"]
    coupling_types = ["AC", "DC"]
    average_types = {"LogPwr": "LOG", "RMS": "RMS", "Voltage": "SCAL"}
    trace_types = ["Average", "MaxHold", "AVG_MH"]

    # Default values:
    test_type = test_types[2]
    coupling = coupling_types[1]
    average_type = list(average_types)[1]
    trace_type = trace_types[2]
    spectrum_range_start = NumericProperty(0)
    spectrum_range_stop = NumericProperty(10000 if test_type == "Default" else 300)
    start_frequency = NumericProperty(1.0)      # NumericProperty binds any changes to the GUI
    stop_frequency = NumericProperty(300.0)     # NumericProperty binds any changes to the GUI
    rbw = 5.1
    vbw = 5.1
    impedance = 50
    attenuation = 10
    reference_level = 10
    y_reference_level = 0

    def build(self):
        Window.bind(size=self.on_window_size)
        return Builder.load_file('Instrument Control GUI.kv')

    def on_window_size(self, window, size):
        width, height = size
        self.is_scrollable = (width > 800 and height < 750) or (width <= 800 and height < 1000)

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

    def show_mismatch_popup(self):
        """Show a popup if the get and set values are mismatched."""
        self.mismatch_dialog = MDDialog(
            title="Set values",
            type="custom",
            content_cls=MismatchDialogContent(),
            buttons=[MDFlatButton(text="OK", on_release=lambda x: self.mismatch_dialog.dismiss()),],
        )
        self.mismatch_dialog.md_bg_color = [1, 1, 1, 1]  # RGBa for white
        self.mismatch_dialog.open()

    def close_dialog(self, *_):
        """Close the popup dialog."""
        self.dialog.dismiss()

    def check_ip_validity(self, *_):
        """Check if the entered IP address is valid."""
        ip_error_label = self.dialog.content_cls.ids.ip_error_label
        ip_input = self.dialog.content_cls.ids.ip_input.text
        pattern = re.compile(regex_ip_pattern)
        if pattern.match(ip_input) is not None:
            connection_error = self.spectrum.connect(ip_input)
            if connection_error is True:
                ip_error_label.text = ""
                self.spectrum_connect(ip_input)
                self.close_dialog()
            else:
                ip_error_label.text = connection_error
        else:
            ip_error_label.text = "Invalid IP address. Please enter a valid one."

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
        self.is_connected = False

    def spectrum_connect(self, ip_input):
        self.root.ids.connection_label.text = f"Connected to {ip_input}"
        self.root.ids.connection_card.md_bg_color = self.color_green
        self.root.ids.connection_button.text = "Disconnect"
        self.is_connected = True

    def check_if_bw_is_0(self, value, caller):
        try:
            value = float(value)
            if value == 0:
                if caller == "rbw":
                    self.root.ids.rbw.text = "AUTO"
                else:
                    self.root.ids.vbw.text = "AUTO"
                return "AUTO"
        except ValueError:
            if caller == "rbw":
                self.root.ids.rbw.text = "AUTO"
            else:
                self.root.ids.vbw.text = "AUTO"
            return "AUTO"
        return value

    @staticmethod
    def check_value_within_limits(value, min_value, max_value, text):
        """Check if GUI value is within the allowed limits; if not, return text."""
        try:
            value = float(value)
            if min_value is not None and value < min_value:
                value = text
            if max_value is not None and value > max_value:
                value = text
            else:
                return value
        except ValueError:
            return value

    def get_gui_values(self):
        self.rbw = self.check_value_within_limits(self.root.ids.rbw.text, 1, None, "AUTO")
        self.rbw = self.check_if_bw_is_0(self.rbw, "rbw")
        self.vbw = self.check_value_within_limits(self.root.ids.vbw.text, 1, None, "AUTO")
        self.vbw = self.check_if_bw_is_0(self.vbw, "vbw")
        self.impedance = self.root.ids.impedance.text
        self.attenuation = self.root.ids.attenuation.text
        self.reference_level = self.root.ids.reference_level.text
        self.y_reference_level = self.root.ids.y_reference_level.text
        self.coupling = self.root.ids.coupling_dropdown.text
        self.average_type = self.root.ids.avg_type_dropdown.text
        self.trace_type = self.root.ids.traces_dropdown.text
        if self.spectrum.use_prints:
            table_to_print = [
                ['start_frequency', self.start_frequency, type(self.start_frequency)],
                ['stop_frequency', self.stop_frequency, type(self.stop_frequency)],
                ['rbw', self.rbw, type(self.rbw)],
                ['vbw', self.vbw, type(self.vbw)],
                ['impedance', self.impedance, type(self.impedance)],
                ['attenuation', self.attenuation, type(self.attenuation)],
                ['reference_level', self.reference_level, type(self.reference_level)],
                ['y_reference_level', self.y_reference_level, type(self.y_reference_level)],
                ['coupling', self.coupling, type(self.coupling)],
                ['average_type', self.average_type, type(self.average_type)],
                ['trace_type', self.trace_type, type(self.trace_type)]
            ]
            print(tabulate(table_to_print, headers=['Parameter', 'Value', 'Type'], tablefmt="fancy_grid"))

    def set_test_type_item(self, item_number):
        """Set the selected item in the Test type dropdown. Should be: "Default", "LF-PLC TX", "LF-PLC RX", "HF-PLC TX", and "HF-PLC RX"""
        self.root.ids.test_type_dropdown.text = self.test_types[item_number]
        match item_number:
            case 0:         # Default
                self.set_start_frequency(1)
                self.set_stop_frequency(10e3)
                self.spectrum_range_start = 0
                self.spectrum_range_stop = 10000
                self.rbw = "AUTO"
                self.vbw = "AUTO"
            case 1:         # LF-PLC TX
                self.set_start_frequency(50)
                self.set_stop_frequency(70)
                self.spectrum_range_start = 0
                self.spectrum_range_stop = 300
                self.rbw = 0.51
                self.vbw = 0.51
            case 2:         # LF-PLC RX
                self.set_start_frequency(1)
                self.set_stop_frequency(300)
                self.spectrum_range_start = 0
                self.spectrum_range_stop = 300
                self.rbw = 5.1
                self.vbw = 5.1
            case 3:         # HF-PLC TX
                self.set_start_frequency(1)
                self.set_stop_frequency(150)
                self.spectrum_range_start = 0
                self.spectrum_range_stop = 300
                self.rbw = 0.68
                self.vbw = 0.68
            case _:         # HF-PLC RX
                self.set_start_frequency(10)
                self.set_stop_frequency(5e3)
                self.spectrum_range_start = 0
                self.spectrum_range_stop = 10000
                self.rbw = 6.8
                self.vbw = 6.8
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
        """Get the values from the GUI and set to Spectrum"""
        self.get_gui_values()
        messages = []
        if self.is_connected:
            messages.append(self.spectrum.get_set_value("impedance", set_value=self.impedance))
            messages.append(self.spectrum.get_set_value("coupling", set_value=self.coupling))
            messages.append(self.spectrum.get_set_value("avg_type", set_value=self.average_type))
            messages.append(self.spectrum.get_set_value("freq_start", set_value=self.start_frequency))
            messages.append(self.spectrum.get_set_value("freq_stop", set_value=self.stop_frequency))
            messages.append(self.spectrum.get_set_value("resolution_bandwidth", set_value=self.rbw))
            messages.append(self.spectrum.get_set_value("video_bandwidth", set_value=self.vbw))
            messages.append(self.spectrum.get_set_value("attenuation", set_value=self.attenuation))
            messages.append(self.spectrum.get_set_value("ref_level", set_value=self.reference_level))
            messages.append(self.spectrum.get_set_value("y_ref_level", set_value=self.y_reference_level))

            # Color mismatches in red:
            for i in range(len(messages)):
                if "mismatch" in messages[i].lower():
                    messages[i] = f"[color=ff0000]{messages[i]}[/color]"
            self.show_mismatch_popup()
            self.mismatch_dialog.content_cls.ids.mismatch_label.text = "\n".join(messages)

    def reset_spectrum(self):
        """Placeholder for reset spectrum method."""
        self.spectrum.reset()

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
