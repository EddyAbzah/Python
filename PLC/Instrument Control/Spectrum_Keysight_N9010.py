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


class KeysightN9010B:
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
        if set_value is not None:
            if isinstance(set_value, str) and set_value.lower() == "auto":
                self.instrument.write(f'{scpi_commands["n9010b_" + measurement]}:AUTO ON')
            else:
                self.instrument.write(f'{scpi_commands["n9010b_" + measurement]} {set_value}')

        # Get the value and convert to float if applicable:
        get_value = self.instrument.query(f'{scpi_commands["n9010b_" + measurement]}?').strip()
        try:
            get_value = float(get_value)
        except ValueError:
            pass  # this is OK... get_value is string
        # Compare the get and the set:
        if set_value is not None and get_value != set_value:
            if self.use_prints:
                print(f'Mismatch in the get / set values!!! {get_value} != {set_value}')
            return f'Mismatch in the get / set values!!! {get_value} != {set_value}'

        # Print and return:
        if self.use_prints:
            print(f'{scpi_syntax[measurement]} = {get_value}')
        return print(f'{scpi_syntax[measurement]} = {get_value}')

    def traces_set(self, measurement, set_value):
        """Set the trace type(s): Average, MaxHold, or both."""
        pass

    def traces_run(self, measurement, set_value):
        """Run (refresh) the traces."""
        pass

    def traces_stop(self, measurement, set_value):
        """Stop the traces."""
        pass


if __name__ == '__main__':
    get_only = False
    spectrum = KeysightN9010B()
    if spectrum.connect("10.20.30.49") is True:
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
            # spectrum.get_set_value("resolution_bandwidth", set_value="AUTO")
            # spectrum.get_set_value("video_bandwidth", set_value="AUTO")
            spectrum.get_set_value("resolution_bandwidth", set_value=510)
            spectrum.get_set_value("video_bandwidth", set_value=5100)
        spectrum.disconnect()
