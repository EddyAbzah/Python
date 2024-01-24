import os

path = r'M:\Users\HW Infrastructure\PLC team\ARC\Temp-Eddy\New_Arc_Detection_Frequency 24_10_21 16_04'
mana = '05 Sliding FFT - MH and AVG Spectrum'
sex = '04 Sliding FFT - MH and AVG Spectrum'
file_name_arr = []
for root, dirs, files in os.walk(path):
    for file in files:
        if mana in file:
            # file_name_arr += [os.path.join(root, '\\', file)]
            file_name_arr += [root + '\\' + file]

for file in file_name_arr:
    os.rename(file, file.replace(mana, sex))
