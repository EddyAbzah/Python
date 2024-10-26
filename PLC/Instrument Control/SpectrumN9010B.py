import pyvisa


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

    def check_mismatch(self, get_value, set_value):
        if set_value is not None and get_value != set_value:
            if self.use_prints:
                print(f'Mismatch in the get / set values!!! {get_value} != {set_value}')
            else:
                return f'Mismatch in the get / set values!!! {get_value} != {set_value}'
        if self.use_prints:
            print(f'Impedance [ohm] = {get_value}')
        else:
            return get_value

    def impedance(self, set_value=None):
        """Get or Set the Impedance [ohm]."""
        if set_value is not None:
            self.instrument.query(f'CORR:IMP?')
        get_value = float(self.instrument.query(f'CORR:IMP?').strip())
        return self.check_mismatch(set_value, get_value)

    def coupling(self, set_value=None):
        """Get or Set the Coupling [AC / DC]."""
        if set_value is not None:
            self.instrument.query(f':INP:COUP?')
        get_value = self.instrument.query(f':INP:COUP?').strip()
        return self.check_mismatch(set_value, get_value)

    def avg_type(self, set_value=None):
        """Get or Set the AvgType [LogPwr / RMS]."""
        if set_value is not None:
            self.instrument.query(f':AVER:TYPE?')
        get_value = self.instrument.query(f':AVER:TYPE?').strip()
        return self.check_mismatch(set_value, get_value)

    def attenuation(self, set_value=None):
        """Get or Set the Attenuation [dB]."""
        if set_value is not None:
            self.instrument.query(f':SENS:POW:RF:ATT?')
        get_value = float(self.instrument.query(f':SENS:POW:RF:ATT?').strip())
        return self.check_mismatch(set_value, get_value)

    def ref_level(self, set_value=None):
        """Get or Set the Reference Level [dBm]."""
        if set_value is not None:
            self.instrument.query(f'DISP:WIND:TRAC:Y:SCAL:RLEV:OFFS?')
        get_value = float(self.instrument.query(f'DISP:WIND:TRAC:Y:SCAL:RLEV:OFFS?').strip())
        return self.check_mismatch(set_value, get_value)

    def y_ref_level(self, set_value=None):
        """Get or Set the Y-axis Reference Level [dBm]."""
        if set_value is not None:
            self.instrument.query(f'DISP:WIND:TRAC:Y:SCAL:RLEV?')
        get_value = float(self.instrument.query(f'DISP:WIND:TRAC:Y:SCAL:RLEV?').strip())
        return self.check_mismatch(set_value, get_value)

    def freq_start(self, set_value=None):
        """Get or Set the Start Frequency [kHz]."""
        if set_value is not None:
            self.instrument.query(f'FREQ:START?')
        get_value = float(self.instrument.query(f'FREQ:START?').strip())
        return self.check_mismatch(set_value, get_value)

    def freq_stop(self, set_value=None):
        """Get or Set the Stop Frequency [kHz]."""
        if set_value is not None:
            self.instrument.query(f'FREQ:STOP?')
        get_value = float(self.instrument.query(f'FREQ:STOP?').strip())
        return self.check_mismatch(set_value, get_value)

    def resolution_bandwidth(self, set_value=None):
        """Get or Set the Resolution Bandwidth [kHz]."""
        if set_value is not None:
            self.instrument.query(f':BAND:RES?')
        get_value = float(self.instrument.query(f':BAND:RES?').strip())
        return self.check_mismatch(set_value, get_value)

    def video_bandwidth(self, set_value=None):
        """Get or Set the Video Bandwidth [kHz]."""
        if set_value is not None:
            self.instrument.query(f':BAND:VID?')
        get_value = float(self.instrument.query(f':BAND:VID?').strip())
        return self.check_mismatch(set_value, get_value)


if __name__ == '__main__':
    spectrum = KeysightN9010B()
    if spectrum.connect("10.20.30.32"):
        spectrum.impedance(set_value=None)
        spectrum.coupling(set_value=None)
        spectrum.avg_type(set_value=None)
        spectrum.attenuation(set_value=None)
        spectrum.ref_level(set_value=None)
        spectrum.y_ref_level(set_value=None)
        spectrum.freq_start(set_value=None)
        spectrum.freq_stop(set_value=None)
        spectrum.resolution_bandwidth(set_value=None)
        spectrum.video_bandwidth(set_value=None)
        spectrum.disconnect()
