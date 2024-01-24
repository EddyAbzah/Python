import os
import sys
import glob
import pandas as pd
import numpy as np
from io import StringIO
from datetime import datetime
from matplotlib import pyplot as plt      # for plt.plot()


# # txt output instead of the console - ATTENTION - if True, there will be no Console output:
output_text = False
path_txt = datetime.now().strftime("%d-%m-%Y %H-%M-%S")
path_txt = f'Terminal Log ({path_txt}).txt'
# # Folders and filters:
path_output = r'C:\Users\eddy.a\Documents\Python Scripts\Stage 1 Improvement\Log Files'
path_logs = r'C:\Users\eddy.a\Documents\Python Scripts\Stage 1 Improvement\Log Files'
path_logs_string_filter = 'Rec'
log_delimiter = ','
log_sample_number_column = 1
log_minimum_line_length = 100


def main():
    log_file_names = []
    files_after_filter = []
    if output_text:
        default_stdout = sys.stdout
        if not os.path.exists(path_output):
            os.makedirs(path_output)
        sys.stdout = open(f'{path_output}/{path_txt}', 'w')
    print(f"Starting Python... Time: {str(datetime.now())[:-7]}")

    list_of_files = glob.glob(f'{path_logs}\\*{path_logs_string_filter}*.log')
    for file_number, file in enumerate(list_of_files):
        print(f'Getting file number {file_number + 1}')
        file_after_filter = []
        with open(file) as temp_file:
            file_before_filter = temp_file.readlines()
        for index_line, line in enumerate(file_before_filter):
            if '@@' in line and len(line) > log_minimum_line_length:
                file_after_filter.append(f'{index_line}{log_delimiter}{line}')
        log_file = pd.read_csv(StringIO('\n'.join(file_after_filter)), sep=log_delimiter,
                               header=None, error_bad_lines=False, skipinitialspace=True).dropna()
        log_line_number = log_file[0].to_numpy().tolist()
        log_sample_number = log_file[log_sample_number_column + 1].to_numpy().tolist()
        diff = np.diff(log_sample_number)
        if max(diff) > 10 or min(diff) < -10:
            bad_samples = np.argwhere(abs(diff) > 10).flatten()
            for sample in bad_samples:
                print(f'There is a jump higher than 10 samples from {log_sample_number[sample]} to {log_sample_number[sample + 1]}')
            cut_log_at = log_line_number[bad_samples[-1] + 1]
            if 'sep=' in file_before_filter[0]:
                del file_before_filter[1:cut_log_at]
            else:
                del file_before_filter[:cut_log_at]
            files_after_filter.append(file_before_filter)
            log_file_names.append(os.path.basename(file))
        else:
            cut_log_at = 0
            print(f'No odd samples found in {os.path.basename(file)}')
        print(f'Record is cut at = {cut_log_at}')
        print()

    for index_file, file_filtered in enumerate(files_after_filter):
        file = open(f'{path_output}/{log_file_names[index_file]}', "w+")
        file.writelines(file_filtered)
        file.close()
    print(f'Python finished... Time: {str(datetime.now())[:-7]}')
    if output_text:
        sys.stdout.close()
        sys.stdout = default_stdout


if __name__ == "__main__":
    main()
