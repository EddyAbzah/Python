import pandas as pd

#print("Enter file path: ", end = "")
#path = input()
path = r"Arc_Power_Paramters"
txt = pd.read_csv(path + "\\CSV.csv")
# Create a Pandas dataframe from the data.
#parameters = {"Bit Map": [0],
#        "PowerDiff": [0],
#        "MaxPhaseShift": [0],
#        "AmplitudeShiftAbsolute": [0],
#        "AmplitudeShiftRatio": [0],
#        "PowerDiffFlag": ["N/A"],
#        "ErgFallFlag": ["N/A"],
#        "PhaseShiftFlag": ["N/A"],
#        "AmpChangeFlag": ["N/A"]
#        }

#df = pd.DataFrame(parameters, columns = ["Bit Map", "PowerDiff","MaxPhaseShift", "AmplitudeShiftAbsolute", "AmplitudeShiftRatio",\
#    "PowerDiffFlag", "ErgFallFlag", "PhaseShiftFlag", "AmpChangeFlag"])

# Create a Pandas Excel writer using XlsxWriter as the engine.
# Convert the dataframe to an XlsxWriter Excel object.
# Close the Pandas Excel writer and output the Excel file.
writer = pd.ExcelWriter(path + "\\Output.xlsx", engine="xlsxwriter")
txt.to_excel(writer, sheet_name="Paramters")
writer.save()