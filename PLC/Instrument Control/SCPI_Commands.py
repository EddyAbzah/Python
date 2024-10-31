scpi_commands = {
    # Spectrum N9010B SCPI Commands:
    "n9010b_impedance": "CORR:IMP",
    "n9010b_coupling": ":INP:COUP",
    "n9010b_avg_type": ":AVER:TYPE",
    "n9010b_attenuation": ":SENS:POW:RF:ATT",
    "n9010b_ref_level": "DISP:WIND:TRAC:Y:SCAL:RLEV:OFFS",
    "n9010b_y_ref_level": "DISP:WIND:TRAC:Y:SCAL:RLEV",
    "n9010b_freq_start": "FREQ:START",
    "n9010b_freq_stop": "FREQ:STOP",
    "n9010b_resolution_bandwidth": ":BAND:RES",
    "n9010b_video_bandwidth": ":BAND:VID",
}

scpi_syntax = {
    "impedance": "Impedance [ohm]",
    "coupling": "Coupling [AC / DC]",
    "avg_type": "Average Type [Log / RMS]",
    "attenuation": "Attenuation [dB]",
    "ref_level": "Reference Level [dBm]",
    "y_ref_level": "Y-axis Reference Level [dBm]",
    "freq_start": "Start Frequency [kHz]",
    "freq_stop": "Stop Frequency [kHz]",
    "resolution_bandwidth": "Resolution Bandwidth [kHz]",
    "video_bandwidth": "Video Bandwidth [kHz]",
}
