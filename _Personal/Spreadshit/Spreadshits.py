import os
import sys
from datetime import datetime
import math
import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


# # txt output instead of the console - ATTENTION - if True, there will be no Console output:
output_text = False
path_txt = datetime.now().strftime("%d-%m-%Y %H-%M-%S")
path_txt = f'Terminal Log ({path_txt}).txt'
# # Chrome action for plotly:
kill_chrome = False
auto_open_chrome_output = False
# # Folders and filters:
path_output = r'C:\Users\Eddy Abzah\Documents\Python\Personal'
path_csv = r'C:\Users\Eddy Abzah\Documents\Python\Personal\Spreadshits'
name_csv_1 = 'Spreadshit'
name_csv_2 = 'Macronutrients'
# # plot name ('.html' ass added later):
plot_name = f'Plot my shitty Spreadshit'
# # divide the output plots into 2D figures:
plot_columns = 1
# # ????:
add_additional_days = 0
skip_macro_titles = 1
# # Add Data Callouts (labels) to the plots:
add_data_labels = False


def main():
    if kill_chrome:
        os.system("taskkill /im chrome.exe /f")
    if output_text:
        default_stdout = sys.stdout
        if not os.path.exists(path_output):
            os.makedirs(path_output)
        sys.stdout = open(f'{path_output}/{path_txt}', 'w')
    print(f"Starting Python... Time: {str(datetime.now())[:-7]}")

    log_file = pd.read_csv(f'{path_csv}/{name_csv_1}.csv')
    csv_titles = list(log_file.columns)
    macros_csv = pd.read_csv(f'{path_csv}/{name_csv_2}.csv', index_col="Macro")
    macro_titles = list(macros_csv.columns)[skip_macro_titles:]
    plot_rows = int(len(csv_titles) / plot_columns)
    all_specs = np.reshape([[{"secondary_y": False}] for x in range(len(csv_titles))],
                           (plot_rows, plot_columns)).tolist()
    fig = make_subplots(subplot_titles=csv_titles[1:], rows=plot_rows, cols=plot_columns,
                        specs=all_specs, shared_xaxes=True)
    list_of_days = []
    list_of_mana = []
    for date_index, date in log_file["Date"].iteritems():
        temp_list = [0 for m in macro_titles[2:]]
        for key, value in enumerate(log_file.loc[date_index]):
            if key < 4:
                continue
            second_list = give_me_the_shit(macros_csv, macro_titles, key, value)
            if isinstance(second_list, str):
                # print(f'{value} is not found in give_me_the_shit(key = {key})')
                list_of_mana.append(value)
            else:
                new_list = [x + y for x, y in zip(temp_list, second_list)]
                temp_list = new_list
        list_of_days.append(temp_list)

    for i_chicko, sex in enumerate(list_of_mana):
        print(f'{i_chicko:03} not found   →   {sex}')

    # for title_index, title in enumerate(csv_titles):
    #     print(f'Plotting the {title}')
    #     fig.add_trace(go.Scatter(y=log_file[title], name=title),
    #                   col=title_index % plot_columns + 1, row=int(title_index / plot_columns) + 1)
    #     print()
    # if add_additional_days != 0:
    #     fig_set_visible_delta = [True] + [False] * 2 * len(csv_titles)
    #     for index_fig, fig in enumerate(csv_titles):
    #         slider_steps = []
    #         for alpha_filter_index, alpha_filter_trace in enumerate(fig_set_visible_delta):
    #             step = dict(args=[{"visible": fig_set_visible_delta}, ],
    #                         label=("%.2f" % alpha_filter_trace))
    #             step["args"][0]["visible"][alpha_filter_index + 1] = True
    #             slider_steps.append(step)
    #         fig.update_layout(sliders=[dict(active=0, pad={"t": 50}, steps=slider_steps, bgcolor="#ffb200",
    #                                         currentvalue=dict(prefix='Alpha filter = ',
    #                                                           xanchor="center", font=dict(size=16)))])
    # if add_data_labels:
    #     labels = [{"plot": 0, "row": 1, "col": 1, "x": 877}, {"plot": 0, "row": 1, "col": 1, "x": 878},
    #               {"plot": 1, "row": 1, "col": 2, "x": 878}, {"plot": 1, "row": 1, "col": 2, "x": 879}]
    #     for label in labels:
    #         x = label["x"]
    #         fig.add_annotation(row=label["row"], col=label["col"], x=x, y=log_file[label["plot"]][x],
    #                            text=f'{x}, {round(log_file[label["plot"]][x], 2)}', showarrow=True, opacity=1)
    # fig.update_layout(title=plot_name, title_font_color="#407294", title_font_size=40,
    #                   legend_title="Foods:", legend_title_font_color="green")
    # plotly.offline.plot(fig, config={'scrollZoom': True}, filename=f'{path_output}/{plot_name}.html',
    #                     auto_open=auto_open_chrome_output)
    #
    # print(f'Python finished... Time: {str(datetime.now())[:-7]}')
    # if output_text:
    #     sys.stdout.close()
    #     sys.stdout = default_stdout


def give_me_the_shit(macros_csv, macro_titles, key, string):
    list_of_string = []
    if "*" in string:
        string_2 = string.split("*")
        times = int(string_2[1].strip())
        string_2 = string_2[0].strip()
    else:
        string_2 = string
        times = 1
    for t in range(times):
        if "+" in string_2:
            for sub_string in string.split("+"):
                list_of_string.append(sub_string.strip())
        else:
            list_of_string.append(string_2)
    list_return = [0 for m in macro_titles[2:]]
    for sub_string in list_of_string:
        second_list = get_dict(macros_csv, macro_titles, key, sub_string)
        if isinstance(second_list, str):
            return second_list
        list_return = [x + y for x, y in zip(list_return, second_list)]
    return list_return


def get_dict(macros_csv, macro_titles, key, string):
    try:
        temp = list(macros_csv.loc[string])[skip_macro_titles:]
    except KeyError:
        temp = string
    return temp


if __name__ == "__main__":
    main()
