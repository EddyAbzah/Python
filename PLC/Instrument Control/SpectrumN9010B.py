import pyvisa
from SCPIcommands import scpi_commands


class KeysightN9010B:
    def __init__(self):
        self.rm = pyvisa.ResourceManager()
        self.instrument = None
        self.use_prints = True

    def connect(self, ip_address):
        """Connect to the spectrum analyzer. Return False is there is no error; otherwise, return the error."""
        resource_string = f'TCPIP0::{ip_address}::INSTR'
        try:
            self.instrument = self.rm.open_resource(resource_string)
            idn = self.instrument.query('*IDN?')
            if self.use_prints:
                print(f'Connected to: {idn}')
            return False        # No error
        except Exception as e:
            if self.use_prints:
                print(f'Connection error: {e}')
            return f'Connection error: {str(e).split(':')[0]}.'

    def disconnect(self):
        """Disconnect from the spectrum analyzer. Return False is there is no error; otherwise, return the error."""
        if self.instrument:
            try:
                self.instrument.close()
                if self.use_prints:
                    print('Disconnected from the instrument.')
                return False
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

    def get_set_value(self, measurement, set_value):
        if set_value is not None:
            self.instrument.write(f'{scpi_commands["n9010b_" + measurement]} {set_value}')
        get_value = self.instrument.query(f'{scpi_commands["n9010b_" + measurement]}?')
        if set_value is not None and get_value != set_value:
            if self.use_prints:
                print(f'Mismatch in the get / set values!!! {get_value} != {set_value}')
            else:
                return f'Mismatch in the get / set values!!! {get_value} != {set_value}'
        if self.use_prints:
            print(f'Impedance [ohm] = {get_value}')
        else:
            return get_value


if __name__ == '__main__':
    spectrum = KeysightN9010B()
    if spectrum.connect("10.20.30.32"):
        spectrum.get_set_value("impedance", set_value=None)
        spectrum.get_set_value("coupling", set_value=None)
        spectrum.get_set_value("avg_type", set_value=None)
        spectrum.get_set_value("attenuation", set_value=None)
        spectrum.get_set_value("ref_level", set_value=None)
        spectrum.get_set_value("y_ref_level", set_value=None)
        spectrum.get_set_value("freq_start", set_value=None)
        spectrum.get_set_value("freq_stop", set_value=None)
        spectrum.get_set_value("resolution_bandwidth", set_value=None)
        spectrum.get_set_value("video_bandwidth", set_value=None)
        spectrum.disconnect()
