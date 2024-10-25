import pyvisa


class KeysightN9010B:
    def __init__(self):
        self.rm = pyvisa.ResourceManager()
        self.instrument = None

    def connect(self, ip_address):
        """Connect to the spectrum analyzer."""
        resource_string = f'TCPIP0::{ip_address}::INSTR'
        try:
            self.instrument = self.rm.open_resource(resource_string)
            idn = self.instrument.query('*IDN?')
            print(f'Connected to: {idn}')
            return False        # No error
        except pyvisa.VisaIOError as e:
            print(f'Connection error: {e}')
            return f'Connection error: {str(e).split(':')[0]}'

    def disconnect(self):
        """Disconnect from the spectrum analyzer."""
        if self.instrument:
            self.instrument.close()
            print('Disconnected from the instrument.')

    def reset(self):
        """Reset the spectrum analyzer."""
        self.instrument.write('*RST')

    def print_all_commands(self):
        """Prints all available SCPI commands."""
        print(self.instrument.query(':SYST:HELP:HEAD?'))

    def get_settings(self):
        """Get the Impedance, Coupling, and AvgType."""
        impedance = float(self.instrument.query(f'CORR:IMP?').strip())
        print(f'{impedance = }')
        coupling = self.instrument.query(f':INP:COUP?').strip()
        print(f'{coupling = }')
        avg_type = self.instrument.query(f':AVER:TYPE?').strip()
        print(f'{avg_type = }')

    def get_levels(self):
        """Get the Attenuation, RefLevel, and YRefLevel."""
        attenuation = float(self.instrument.query(f':SENS:POW:RF:ATT?').strip())
        print(f'{attenuation = }')
        ref_level = float(self.instrument.query('DISP:WIND:TRAC:Y:SCAL:RLEV:OFFS?').strip())
        print(f'{ref_level = }')
        y_ref_level = float(self.instrument.query('DISP:WIND:TRAC:Y:SCAL:RLEV?').strip())
        print(f'{y_ref_level = }')

    def get_frequencies(self):
        """Get the StartFreq, StopFreq, and BW."""
        freq_start = float(self.instrument.query(f'FREQ:START?').strip())
        print(f'{freq_start = }')
        freq_stop = float(self.instrument.query(f'FREQ:STOP?').strip())
        print(f'{freq_stop = }')
        freq_center = float(self.instrument.query(f'FREQ:CENTER?').strip())
        print(f'{freq_center = }')
        resolution_bandwidth = float(self.instrument.query(f':BAND:RES?').strip())
        print(f'{resolution_bandwidth = }')
        video_bandwidth = float(self.instrument.query(f':BAND:VID?').strip())
        print(f'{video_bandwidth = }')


if __name__ == '__main__':
    spectrum = KeysightN9010B()
    if spectrum.connect("10.20.30.32"):
        spectrum.get_settings()
        spectrum.get_levels()
        spectrum.get_frequencies()
        spectrum.disconnect()
