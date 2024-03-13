import os
import pandas as pd

def compare_and_write_excel(file1, file2, output_file):
    dfs = [pd.read_excel(file1, sheet_name=None), pd.read_excel(file2, sheet_name=None)]
    writer = pd.ExcelWriter(output_file, engine='xlsxwriter', mode='w')
    for df in dfs:
        for sheet_name in list(df):
            if sheet_name == 'Config':
                df.pop(sheet_name)
            else:
                df[''.join([c for c in sheet_name if not c.isdigit()])] = df.pop(sheet_name)

    for sheet_name in dfs[0].keys():
        print(f'{sheet_name = }')
        if sheet_name in dfs[1]:
            df1_sheet = dfs[0][sheet_name]
            df2_sheet = dfs[1][sheet_name]
            min_index = df1_sheet.index if len(df1_sheet.index) < len(df2_sheet.index) else df2_sheet.index
            min_columns = df1_sheet.columns if len(df1_sheet.columns) < len(df2_sheet.columns) else df2_sheet.columns
            comparison_result = pd.DataFrame(index=min_index, columns=min_columns)

            for i in range(len(min_index)):
                for j in range(len(min_columns)):
                    cell1 = df1_sheet.iloc[i, j]
                    cell2 = df2_sheet.iloc[i, j]

                    # if pd.isna(cell1) or pd.isna(cell2):
                    #     comparison_result.iloc[i, j] = "_"
                    # elif isinstance(cell1, str) or isinstance(cell2, str):
                    #     comparison_result.iloc[i, j] = f'{cell1} {cell2}'
                    # else:
                    #     comparison_result.iloc[i, j] = pd.to_numeric(cell1, errors='coerce') - pd.to_numeric(cell2, errors='coerce')

                    if pd.isna(cell1) and pd.isna(cell2):
                        comparison_result.iloc[i, j] = ""
                    elif pd.isna(cell1):
                        comparison_result.iloc[i, j] = "C1 empty"
                    elif pd.isna(cell2):
                        comparison_result.iloc[i, j] = "C2 empty"
                    elif isinstance(cell1, str) and isinstance(cell1, str):
                        if cell1 == cell2:
                            comparison_result.iloc[i, j] = cell1
                        else:
                            comparison_result.iloc[i, j] = f'C1={cell1} _ C2={cell2}'
                    else:
                        comparison_result.iloc[i, j] = pd.to_numeric(cell1, errors='coerce') - pd.to_numeric(cell2, errors='coerce')

            comparison_result.to_excel(writer, sheet_name=sheet_name, index=False)
        else:
            print(f'sheet_name {sheet_name} not found in file_2')

    writer.close()
    print(f"Comparison results saved to '{output_file}'")


if __name__ == "__main__":
    folder = r"M:\Users\HW Infrastructure\PLC team\ARC\Temp-Eddy\Optimizer Automation Sanity Tests" + "\\"
    a, b = "25C", "85C"
    file_1 = folder + f"Sanity Test 06 {a}.xlsx"
    file_2 = folder + f"Sanity Test 06 {b}.xlsx"
    file_out = folder + f"Sanity Test 06 - {a} vs {b}.xlsx"

    compare_and_write_excel(file_1, file_2, file_out)
    os.startfile(file_out)
