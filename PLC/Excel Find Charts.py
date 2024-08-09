import openpyxl
from openpyxl.drawing.image import Image


def get_chart_details(file_path):
    # Load the workbook and iterate through each worksheet
    workbook = openpyxl.load_workbook(file_path)
    chart_details = []

    for sheet_name in workbook.sheetnames:
        worksheet = workbook[sheet_name]
        for drawing in worksheet._charts:
            chart = drawing
            pos = chart.anchor._from  # Position details
            size = chart.anchor.to  # Size details

            chart_info = {
                'Sheet': sheet_name,
                'Chart': chart.title,
                'Legend': chart.series[0].title.value,
                'Start column': pos.col,
                'Stop column': size.col,
                'Start Row': pos.row,
                'Stop Row': size.row,
                'Width': size.col - pos.col,
                'Height': size.row - pos.row
            }
            chart_details.append(chart_info)

    return chart_details


# Example usage
file_path = r"C:\Users\eddy.a\Downloads\Automation Console 2\Arcs 02\Full_Arc_Test-05_08_24-17_10\Test 06 with CosPhi.xlsx"
chart_details = get_chart_details(file_path)

print("\t".join(chart_details[0].keys()))
for chart in chart_details:
    print("\t".join([str(v) for v in chart.values()]))
