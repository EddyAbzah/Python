import os
import pandas as pd


folder_path = r""
output_excel = r"Recursive File Lister.xlsx"
max_files = 10  # Limit to 10 largest files


def get_largest_files(path):
    """Returns the 10 largest files in the given folder."""
    files = [(f, os.path.getsize(os.path.join(path, f))) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    files.sort(key=lambda x: x[1], reverse=True)  # Sort by file size
    return [f"{name} ({size/1e6:.2f} MB)" for name, size in files[:10]]  # Convert size to MB


base_folder_name = os.path.basename(folder_path)  # Root folder name
data = []
for root, _, _ in os.walk(folder_path): # Walk through all subfolders recursively
    if root == folder_path:
        folder_levels = [base_folder_name]
    else:
        folder_levels = [base_folder_name] + root[len(folder_path):].strip(os.sep).split(os.sep)  # Get folder levels
    row_data = {f"Folder {i + 1:02}": f for i, f in enumerate(folder_levels)}
    row_data.update({f"File {i + 1:02}": f for i, f in enumerate(get_largest_files(root))})
    data.append(row_data)


df = pd.DataFrame(data)
new_column_order = [col for col in df.columns if 'Folder' in col] + [col for col in df.columns if 'File' in col]
df = df[new_column_order]

df.to_excel(output_excel, index=False)
print(f"Excel file saved as {output_excel}")
