"""
Script to control the Keysight N9010A or N9010B.
If you don't have "Keysight IO Library" installed, install via pip "PyVISA-py".

Commands:
Impedance: choose between 50ohm or 75ohm.
Coupling: choose AC or DC.
Average Type: 3 options: LogPwr (LOG), RMS, Voltage (SCAL).
Attenuation: enter number in dB; default = 10dB.
Reference Level: enter number in dBm; default = 0dBm. Set to 10 to offset for external 100ohm, or to 20 to offset for external 500ohm.
Y-axis Reference Level: enter number in dBm; default = 0dBm. This only affects the visuals, not the measurements.
Start Frequency: enter number in kHz.
Stop Frequency: enter number in kHz.
Resolution Bandwidth: enter number in kHz.
Video Bandwidth: enter number in kHz.
"""


# install PyVISA-py via pip
import pyvisa
from SCPI_Commands import scpi_commands, scpi_syntax


default_avg_count = 100
enable_annotation = True
full_screen = True


# public virtual Image Screenshot()
# {
#     Image image;
#     WebRequest request;
#     Stream stream;
#     WebResponse response;
# 
#     //Taking picture from the browser server - more reliable:
#     request = _isNewImagePath ? WebRequest.Create("http://" + (IO as SocketAdapter).IP + "/WebInstrumentAspx/ScreenImageHandler.ashx") : WebRequest.Create("http://" + (IO as SocketAdapter).IP + "/Agilent.SA.WebInstrument/Screen.png");
# 
#     response = request.GetResponse();
#     stream = response.GetResponseStream();
#     image = Image.FromStream(stream);
# 
#     return image;
# }


class KeysightN9010B:
    ip_address = ""      # default IP address for the Spectrum

    def __init__(self):
        self.rm = pyvisa.ResourceManager()
        self.instrument = None
        self.use_prints = True

    def connect(self, ip_address):
        """Connect to the spectrum analyzer. Return True is there is no error; otherwise, return the error."""
        resource_string = f'TCPIP0::{ip_address}::INSTR'
        try:
            self.instrument = self.rm.open_resource(resource_string)
            idn = self.instrument.query('*IDN?')
            if self.use_prints:
                print(f'Connected to: {idn.replace(",", ", ")}')
            return True        # No error
        except Exception as e:
            if self.use_prints:
                print(f'Connection error: {e}')
            return f'Connection error: {str(e).split(':')[0]}.'

    def disconnect(self):
        """Disconnect from the spectrum analyzer. Return True is there is no error; otherwise, return the error."""
        if self.instrument:
            try:
                self.instrument.close()
                if self.use_prints:
                    print('Disconnected from the instrument.')
                return True
            except Exception as e:
                if self.use_prints:
                    print(f'Connection error: {e}')
                return f'Connection error: {str(e).split(':')[0]}.'
        else:
            return f'There is no instrument connected.'

    def reset(self):
        """Reset the spectrum analyzer. Return True is reset is completed."""
        self.instrument.write('*RST')
        if self.use_prints:
            print('Spectrum is reset.')
        return True

    def clear_errors(self):
        """Reset the spectrum analyzer. Return True is reset is completed."""
        self.instrument.write('*CLS')
        if self.use_prints:
            print('Spectrum has been cleared.')
        return True

    def set_basic_parameters(self):
        """Set basic parameters like annotations or full screen mode; the parameters are at the top of the code"""
        self.instrument.write(f':DISP:FSCR {"ON" if full_screen else "OFF"}')
        self.instrument.write(f':DISP:ANN:TRAC {"ON" if enable_annotation else "OFF"}')
        if self.use_prints:
            print(f'Spectrum full screen is set to {full_screen = }.')
            print(f'Spectrum annotations is set to {enable_annotation = }.')
        return True

    def print_all_commands(self):
        """Prints all available SCPI commands."""
        all_commands = self.instrument.query(':SYST:HELP:HEAD?')
        if self.use_prints:
            print(f"All spectrum commands:\n{all_commands}")
        return all_commands

    def get_set_value(self, measurement, set_value=None):
        """Get the SCPI command from the SCPI_Commands.py file, and send to the Spectrum.
        Return the value as string if set correctly; otherwise, return the error."""
        # Set value to Spectrum if not None:
        auto = isinstance(set_value, str) and set_value.lower() == "auto"
        if set_value is not None:
            if auto:
                self.instrument.write(f'{scpi_commands["n9010b_" + measurement]}:AUTO ON')
            else:
                if "freq" in measurement or "bandwidth" in measurement:
                    set_value *= 1000
                self.instrument.write(f'{scpi_commands["n9010b_" + measurement]} {set_value}')

        # Get the value and convert to float if applicable:
        get_value = self.instrument.query(f'{scpi_commands["n9010b_" + measurement]}?').strip()

        try:
            set_value = float(set_value)
            if "freq" in measurement or "bandwidth" in measurement:
                set_value /= 1000
        except (ValueError, TypeError):
            pass  # this is OK... get_value is either string (ValueError) or None (TypeError)
        try:
            get_value = float(get_value)
            if "freq" in measurement or "bandwidth" in measurement:
                get_value /= 1000
        except ValueError:
            pass  # this is OK... get_value is string

        # Compare the get and the set:
        if set_value is not None and get_value != set_value and not auto:
            message = f'Mismatch in the get / set values!!! {get_value} != {set_value}'
            if self.use_prints:
                print(message)
            return message

        # Print and return:
        message = f'{scpi_syntax[measurement]} = {get_value}'
        if auto:
            message = message + " (automatically set)"
        if self.use_prints:
            print(message)
        return message

    def traces_set(self, indexes, modes):
        """Set the trace type, label, and display
        Parameters:
            indexes: int or list(int) - Trace number to configure (1 to 6).
            modes: str or list(str) - Trace mode ("WRITE", "AVER", "MAXHOLD", "MINHOLD")."""
        if not isinstance(indexes, list):
            indexes = [indexes]
        if not isinstance(modes, list):
            modes = [modes]

        if "AVER" in modes:
            self.instrument.write(f"SENS:AVER:STATE OFF")
            self.instrument.write(f"SENS:AVER:STATE ON")
            self.instrument.write(f"SENS:AVER:COUNT {default_avg_count}")

        for index, mode in zip(indexes, modes):
            if mode != "AVER":
                # Glitch in Spectrum = changing mode doesn't change the label, so in the next iteration the mode doesn't change:
                current_mode = self.instrument.query(f":TRACE{index}:MODE?").strip()
                if mode == current_mode:
                    temp_mode = mode.replace("MAXH", "TEMP").replace("MINH", "MAXH").replace("TEMP", "MINH")
                    self.instrument.write(f":TRACE{index}:MODE {temp_mode}")
                # End of Glitch handle
                self.instrument.write(f":TRACE{index}:MODE {mode}")
            if self.use_prints:
                print(f"Trace {index} is set to mode {mode}")
        self.traces_run(indexes)

    def traces_run(self, indexes):
        """Run (refresh) the traces."""
        self.instrument.write(f"INIT:CONT ON")
        if not isinstance(indexes, list):
            indexes = [indexes]
        for index in indexes:
            self.instrument.write(f"TRAC{index}:UPD ON")
            self.instrument.write(f"TRAC{index}:DISP ON")

    def traces_stop(self, indexes):
        """Stop the traces."""
        if not isinstance(indexes, list):
            indexes = [indexes]
        for index in indexes:
            self.instrument.write(f"TRAC{index}:UPD OFF")


if __name__ == '__main__':
    get_only = True
    traces_run = False
    traces_stop = False
    spectrum = KeysightN9010B()
    if spectrum.connect(spectrum.ip_address) is True:

        spectrum.clear_errors()
        spectrum.set_basic_parameters()

        if traces_run:
            spectrum.traces_set([1, 2], ["AVER", "MAXH"])
            spectrum.traces_run([1, 2])
        elif traces_stop:
            spectrum.traces_stop([1, 2])

        if get_only:
            spectrum.get_set_value("impedance")
            spectrum.get_set_value("coupling")
            spectrum.get_set_value("avg_type")
            spectrum.get_set_value("attenuation")
            spectrum.get_set_value("ref_level")
            spectrum.get_set_value("y_ref_level")
            spectrum.get_set_value("freq_start")
            spectrum.get_set_value("freq_stop")
            spectrum.get_set_value("resolution_bandwidth")
            spectrum.get_set_value("video_bandwidth")
        else:
            spectrum.get_set_value("impedance", set_value=50)
            spectrum.get_set_value("coupling", set_value="DC")
            spectrum.get_set_value("avg_type", set_value="RMS")
            spectrum.get_set_value("attenuation", set_value=10)
            spectrum.get_set_value("ref_level", set_value=0)
            spectrum.get_set_value("y_ref_level", set_value=0)
            spectrum.get_set_value("freq_start", set_value=10e3)
            spectrum.get_set_value("freq_stop", set_value=300e3)
            spectrum.get_set_value("resolution_bandwidth", set_value=510)   # or "AUTO"
            spectrum.get_set_value("video_bandwidth", set_value=5100)       # or "AUTO"
        spectrum.disconnect()
