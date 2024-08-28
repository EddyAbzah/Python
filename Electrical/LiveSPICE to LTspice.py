"""This script is for converting LiveSPICE files to LTspice"""

path_folder = r"C:\Users\eddy.a\OneDrive - SolarEdge\Documents\Yuvi et Eddy\Pedals - future\Way Huge - Swollen Pickle MkII\LTspice" + "\\"
path_in = "Way Huge - Swollen Pickle MkII 01" + ".schx"
path_out = "Way Huge - Swollen Pickle MkII 01" + ".asc"
txt_out = ["Version 4"]

dic = {
    "Circuit.Wire": "WIRE",
    "Circuit.Ground": "FLAG",
    "Circuit.Rail": "FLAG",
    "Circuit.Input": "voltage",
    "Circuit.Capacitor": "cap",
    "Circuit.Resistor": "res",
    "Circuit.Speaker": "res",   # if speaker resistance value is replaced from '∞' to '10Meg' => .replace('∞', '10Meg')
    "Circuit.Diode": "diode",
    "Circuit.VariableResistor": "potentiometer_standard",
    "Circuit.Potentiometer": "potentiometer_standard",
    "Circuit.BipolarJunctionTransistor": "BJT",
    "Circuit.IdealOpAmp": "OpAmps\\\\UniversalOpAmp2"
}
# inverted_dict = dict(map(reversed, dic.items()))


with open(path_folder + path_in, encoding='utf-8') as f:
    lines = f.readlines()
print(f"SCHX is in - len(lines) = {len(lines)}")
print()
symbol_position_x = 0
symbol_position_y = 656

for key, value in dic.items():
    print(f"key = {key}, value = {value}")
    for index, line in enumerate(lines):
        if key in line:

            if value == "WIRE":
                position = line[line.find("A=") + 3:line.find(" B=") - 1] + ',' + line[line.find("B=") + 3:line.find(" />") - 1]
                x1, y1, x2, y2 = [((int(n) * 2) // 16) * 16 for n in position.split(',')]
                txt_out.append(f"{value} {x1} {y1} {x2} {y2}")
            elif value == "FLAG":
                line_before = lines[index - 1]
                position = line_before[line_before.find("Position=") + 10:line_before.find(">") - 1]
                x1, y1 = [((int(n) * 2) // 16) * 16 for n in position.split(',')]
                last_char = '0' if key == "Circuit.Ground" else line[line.find("Voltage=") + 9:line.find(" Name=") - 1]
                txt_out.append(f"{value} {x1} {y1} {last_char}")

            elif value == "voltage":
                txt_out.append(f"SYMBOL {value} {symbol_position_x} {symbol_position_y} R0")
                txt_out.append("SYMATTR InstName " + line[line.find("Name=") + 6:line.find(" Description=") - 1])

            elif value == "BJT":
                txt_out.append(f"SYMBOL {line[line.find(' Type=') + 7:line.find(' IS=') - 1].lower()} {symbol_position_x} {symbol_position_y} R0")
                txt_out.append("SYMATTR InstName " + line[line.find("Name=") + 6:line.find(" PartNumber=") - 1])

            else:
                txt_out.append(f"SYMBOL {value} {symbol_position_x} {symbol_position_y} R0")
                line_split = line[line.find("Name=") + 6:]
                txt_out.append("SYMATTR InstName " + line_split[:line_split.find('\u0022')])
                if value == "diode":
                    line_split = line[line.find("PartNumber=") + 12:]
                    line_split = line_split[:line_split.find('\u0022')]
                    if line_split == "Red" or line_split == "Green" or line_split == "Blue":
                        txt_out[-2] = txt_out[-2].replace("diode", "LED")
                    elif line_split != "Si" or line_split != "Ge":
                        txt_out.append(f"SYMATTR Value {line_split}")
                    else:
                        txt_out.append("SYMATTR Value D")
                elif value == "potentiometer_standard":
                    line_split = f"SYMATTR Value Rtot={line[line.find('ance=') + 6:line.find(' Wipe=') - 2].replace(' ', '').replace('μ', 'u')} wiper="
                    txt_out.append(line_split + line[line.find('Wipe=') + 6:line.find(' Sweep=') - 1].replace(' ', '').replace('μ', 'u'))
                    txt_out.append('SYMATTR SpiceLine ""')
                else:
                    txt_out.append(f"SYMATTR Value {line[line.find('ance=') + 6:line.find(' Name=') - 2].replace(' ', '').replace('μ', 'u').replace('∞', '10Meg')}")

            if value != "WIRE" or value != "FLAG":
                if symbol_position_x > 656:
                    symbol_position_x = 0
                    symbol_position_y += 16 * 8
                else:
                    if value == "potentiometer_standard" or "OpAmps\\\\UniversalOpAmp2":
                        symbol_position_x += 16 * 8
                    else:
                        symbol_position_x += 16 * 4

with open(path_folder + path_out, 'w', encoding="utf-8") as f:
    for line in txt_out:
        f.write(f"{line}\n")
