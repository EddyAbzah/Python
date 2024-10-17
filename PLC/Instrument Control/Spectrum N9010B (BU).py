import pyvisa


class KeysightN9010B:
    def __init__(self, ip_address):
        self.ip_address = ip_address
        self.rm = pyvisa.ResourceManager()
        self.instrument = None

    def connect(self):
        """Connect to the spectrum analyzer."""
        resource_string = f'TCPIP0::{self.ip_address}::INSTR'
        try:
            self.instrument = self.rm.open_resource(resource_string)
            idn = self.instrument.query('*IDN?')
            print(f'Connected to: {idn}')
        except pyvisa.VisaIOError as e:
            print(f'Connection error: {e}')
            raise

    def disconnect(self):
        """Disconnect from the spectrum analyzer."""
        if self.instrument:
            self.instrument.close()
            print('Disconnected from the instrument.')

    def set_center_frequency(self, frequency):
        """Set the center frequency in Hz."""
        self.instrument.write(f'FREQ:CENTER {frequency}')

    def set_frequency_span(self, span):
        """Set the frequency span in Hz."""
        self.instrument.write(f'FREQ:SPAN {span}')

    def turn_continuous_mode(self, on=True):
        """Turn continuous mode on or off."""
        self.instrument.write('INIT:CONT ' + ('ON' if on else 'OFF'))

    def get_marker_value(self):
        """Get the value of the current marker."""
        return self.instrument.query('CALC:MARK:FUNC:POIN:ALL?')

    def measure_peak(self):
        """Set the marker to peak and return the marker value."""
        self.instrument.write('CALC:MARK:FORM PEAK')
        return self.get_marker_value()

    def get_spectrum_data(self):
        """Retrieve spectrum data."""
        self.instrument.write('FORM:DATA REAL,32')
        data = self.instrument.query('TRAC:DATA? TRACE1')
        return [float(d) for d in data.split(',')]

    def reset(self):
        """Reset the spectrum analyzer."""
        self.instrument.write('*RST')


if __name__ == '__main__':
    analyzer = KeysightN9010B("10.20.30.32")
    try:
        analyzer.connect()
        analyzer.set_center_frequency(1e9)  # Set to 1 GHz
        analyzer.set_frequency_span(100e6)   # Set span to 100 MHz
        analyzer.turn_continuous_mode(False)

        peak_value = analyzer.measure_peak()
        print(f'Peak Marker Value: {peak_value}')

        spectrum_data = analyzer.get_spectrum_data()
        print(f'Spectrum Data: {spectrum_data}')

    finally:
        analyzer.disconnect()
