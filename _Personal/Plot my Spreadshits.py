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
path_output = r'C:\Users\eddy.a\Documents\Python Scripts\Personal'
path_csv = r'C:\Users\eddy.a\Documents\Python Scripts\Personal'
path_csv_name = 'Spreadshit'
csv_titles = ['Baseline 1', 'Baseline 2', 'Drink', 'Meal 1', 'Meal 2', 'Meal 3', 'Meal 4', 'Meal 5']
# # plot name ('.html' ass added later):
plot_name = f'Plot my shitty Spreadshit'
# # divide the output plots into 2D figures:
plot_columns = 1
# # Energy Rise and Current Drop parameters:
add_additional_days = 0
# # Add Data Callouts (labels) to the plots:
add_data_labels = False


def macros(protein=0, sugar=0, fats=0, white_meat=0, red_meat=0, fish=0, oil=0, milk=0, salts=0, caffeine=0):
    macros_list = [protein, sugar, fats, white_meat, red_meat, fish, oil, milk, salts, caffeine]
    return macros_list


def main():
    if kill_chrome:
        os.system("taskkill /im chrome.exe /f")
    if output_text:
        default_stdout = sys.stdout
        if not os.path.exists(path_output):
            os.makedirs(path_output)
        sys.stdout = open(f'{path_output}/{path_txt}', 'w')
    print(f"Starting Python... Time: {str(datetime.now())[:-7]}")

    plot_rows = int(len(csv_titles) / plot_columns)
    all_specs = np.reshape([[{"secondary_y": False}] for x in range(len(csv_titles))],
                           (plot_rows, plot_columns)).tolist()
    fig = make_subplots(subplot_titles=csv_titles, rows=plot_rows, cols=plot_columns,
                        specs=all_specs, shared_xaxes=True)

    log_file = pd.read_csv(f'{path_csv}/{path_csv_name}.csv')
    list_of_days = []
    list_of_mana = []
    for date_index, date in log_file["Date"].items():
        first_list = macros()
        for key, value in enumerate(log_file.loc[date_index]):
            if key < 4:
                continue
            second_list = give_me_the_shit(key, value)
            if isinstance(second_list, str):
                # print(f'{value} is not found in give_me_the_shit(key = {key})')
                list_of_mana.append(value)
            else:
                new_list = [x + y for x, y in zip(first_list, second_list)]
                first_list = new_list
        list_of_days.append(new_list)

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


def give_me_the_shit(key, string):
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
    list_return = macros()
    for sub_string in list_of_string:
        second_list = get_dict(key, sub_string)
        if isinstance(second_list, str):
            return second_list
        list_return = [x + y for x, y in zip(list_return, second_list)]
    return list_return


def get_dict(key, string):
    pee_dict = {
        "water":
            macros(protein=0, sugar=0, fats=0, white_meat=0, red_meat=0, fish=0, oil=0, milk=0, salts=0, caffeine=0),
        "black coffee":
            macros(protein=0, sugar=0, fats=0, white_meat=0, red_meat=0, fish=0, oil=0, milk=0, salts=0, caffeine=5),
        "coffee with milk":
            macros(protein=0, sugar=1, fats=0, white_meat=0, red_meat=0, fish=0, oil=0, milk=2, salts=0, caffeine=3),
        "pomegranate soda":
            macros(protein=0, sugar=1, fats=0, white_meat=0, red_meat=0, fish=0, oil=0, milk=0, salts=0, caffeine=0)
    }

    shit_dict = {
        "air":
            macros(protein=0, sugar=0, fats=0, white_meat=0, red_meat=0, fish=0, oil=0, milk=0, salts=0, caffeine=0),
        "pear":
            macros(protein=0, sugar=1, fats=0, white_meat=0, red_meat=0, fish=0, oil=0, milk=0, salts=0, caffeine=0),
        "gamba":
            macros(protein=0, sugar=1, fats=0, white_meat=0, red_meat=0, fish=0, oil=0, milk=0, salts=0, caffeine=0),
        "apple":
            macros(protein=0, sugar=1, fats=0, white_meat=0, red_meat=0, fish=0, oil=0, milk=0, salts=0, caffeine=0),
        "peach":
            macros(protein=0, sugar=3, fats=0, white_meat=0, red_meat=0, fish=0, oil=0, milk=0, salts=0, caffeine=0),
        "nectarine":
            macros(protein=0, sugar=3, fats=0, white_meat=0, red_meat=0, fish=0, oil=0, milk=0, salts=0, caffeine=0),
        "plum":
            macros(protein=0, sugar=3, fats=0, white_meat=0, red_meat=0, fish=0, oil=0, milk=0, salts=0, caffeine=0),
        "banana":
            macros(protein=0, sugar=1, fats=0, white_meat=0, red_meat=0, fish=0, oil=0, milk=0, salts=0, caffeine=0),
        "grapes":
            macros(protein=0, sugar=1, fats=0, white_meat=0, red_meat=0, fish=0, oil=0, milk=0, salts=0, caffeine=0),
        "yoplea with granola":
            macros(protein=0, sugar=3, fats=0, white_meat=0, red_meat=0, fish=0, oil=0, milk=3, salts=0, caffeine=0),
        "greko shwarma":
            macros(protein=4, sugar=0, fats=4, white_meat=2, red_meat=2, fish=0, oil=1, milk=0, salts=2, caffeine=0),
        "greko scurdelia":
            macros(protein=0, sugar=2, fats=2, white_meat=0, red_meat=0, fish=0, oil=1, milk=0, salts=1, caffeine=0),
        "greko":
            macros(protein=4, sugar=2, fats=6, white_meat=2, red_meat=2, fish=0, oil=2, milk=0, salts=3, caffeine=0),
        "gulasch with beans":
            macros(protein=5, sugar=0, fats=4, white_meat=2, red_meat=2, fish=0, oil=2, milk=0, salts=0, caffeine=0),
        "schips pasta":
            macros(protein=3, sugar=0, fats=6, white_meat=3, red_meat=0, fish=0, oil=4, milk=0, salts=0, caffeine=0),
        "pasta with schnitzel":
            macros(protein=4, sugar=2, fats=0, white_meat=4, red_meat=0, fish=0, oil=0, milk=0, salts=0, caffeine=0),
        "haluz":
            macros(protein=2, sugar=2, fats=4, white_meat=0, red_meat=0, fish=0, oil=3, milk=4, salts=0, caffeine=0),
        "mataz":
            macros(protein=2, sugar=2, fats=2, white_meat=0, red_meat=0, fish=0, oil=0, milk=4, salts=0, caffeine=0),
        "qulaq":
            macros(protein=2, sugar=2, fats=2, white_meat=0, red_meat=0, fish=0, oil=2, milk=4, salts=0, caffeine=0),
        "bamba":
            macros(protein=4, sugar=3, fats=4, white_meat=0, red_meat=0, fish=0, oil=0, milk=0, salts=0, caffeine=0),
        "pizza":
            macros(protein=3, sugar=1, fats=2, white_meat=0, red_meat=0, fish=0, oil=1, milk=2, salts=0, caffeine=0),
        "chocolate small":
            macros(protein=0, sugar=2, fats=1, white_meat=0, red_meat=0, fish=0, oil=0, milk=1, salts=0, caffeine=0),
        "chocolate":
            macros(protein=0, sugar=4, fats=2, white_meat=0, red_meat=0, fish=0, oil=0, milk=2, salts=0, caffeine=0),
        "chocolate large":
            macros(protein=0, sugar=6, fats=3, white_meat=0, red_meat=0, fish=0, oil=0, milk=3, salts=0, caffeine=0),
        "tarabin":
            macros(protein=4, sugar=3, fats=4, white_meat=4, red_meat=0, fish=0, oil=4, milk=0, salts=1, caffeine=0),
        "abo hasan":
            macros(protein=3, sugar=1, fats=2, white_meat=3, red_meat=0, fish=0, oil=2, milk=0, salts=0, caffeine=0),
        "avocado small":
            macros(protein=0, sugar=0, fats=3, white_meat=0, red_meat=0, fish=0, oil=0, milk=0, salts=0, caffeine=0),
        "avocado":
            macros(protein=0, sugar=0, fats=6, white_meat=0, red_meat=0, fish=0, oil=0, milk=0, salts=0, caffeine=0),
        "labane":
            macros(protein=2, sugar=2, fats=1, white_meat=0, red_meat=0, fish=0, oil=0, milk=5, salts=0, caffeine=0),
        "tari bari":
            macros(protein=3, sugar=0, fats=1, white_meat=0, red_meat=0, fish=0, oil=0, milk=0, salts=0, caffeine=0),
        "bamba small":
            macros(protein=2, sugar=1, fats=2, white_meat=0, red_meat=0, fish=0, oil=0, milk=0, salts=0, caffeine=0),
        "shwarma":
            macros(protein=3, sugar=0, fats=3, white_meat=2, red_meat=3, fish=0, oil=0, milk=0, salts=2, caffeine=0),
        "ycs":
            macros(protein=2, sugar=1, fats=5, white_meat=0, red_meat=0, fish=0, oil=0, milk=4, salts=0, caffeine=0),
        "ycs small":
            macros(protein=1, sugar=1, fats=2, white_meat=0, red_meat=0, fish=0, oil=0, milk=2, salts=0, caffeine=0),
        "sima's pizza":
            macros(protein=2, sugar=1, fats=5, white_meat=0, red_meat=0, fish=0, oil=0, milk=4, salts=2, caffeine=0),
        "tapoo":
            macros(protein=1, sugar=1, fats=3, white_meat=0, red_meat=0, fish=0, oil=0, milk=0, salts=4, caffeine=0),
        "popcorn":
            macros(protein=0, sugar=0, fats=2, white_meat=0, red_meat=0, fish=0, oil=0, milk=0, salts=4, caffeine=0),
        "cheetos":
            macros(protein=0, sugar=0, fats=2, white_meat=0, red_meat=0, fish=0, oil=0, milk=0, salts=4, caffeine=0),
        "bisley":
            macros(protein=0, sugar=0, fats=2, white_meat=0, red_meat=0, fish=0, oil=0, milk=0, salts=4, caffeine=0),
        "rice with kuba":
            macros(protein=1, sugar=0, fats=2, white_meat=0, red_meat=3, fish=0, oil=0, milk=0, salts=1, caffeine=0),
        "sugar medium":
            macros(protein=0, sugar=3, fats=0, white_meat=0, red_meat=0, fish=0, oil=0, milk=0, salts=0, caffeine=0),
        "al haesh":
            macros(protein=5, sugar=0, fats=3, white_meat=3, red_meat=3, fish=0, oil=1, milk=0, salts=3, caffeine=0),
        "tuna sandwich":
            macros(protein=5, sugar=0, fats=2, white_meat=0, red_meat=0, fish=2, oil=0, milk=0, salts=1, caffeine=0),
        "yoplait":
            macros(protein=3, sugar=1, fats=0, white_meat=0, red_meat=0, fish=0, oil=0, milk=4, salts=0, caffeine=0),
        "yoplait tut":
            macros(protein=3, sugar=2, fats=0, white_meat=0, red_meat=0, fish=0, oil=0, milk=4, salts=0, caffeine=0),
        "homus":
            macros(protein=4, sugar=0, fats=0, white_meat=0, red_meat=0, fish=0, oil=0, milk=0, salts=0, caffeine=0),
        "homus chips":
            macros(protein=4, sugar=0, fats=2, white_meat=0, red_meat=0, fish=0, oil=2, milk=0, salts=0, caffeine=0),
        "fish":
            macros(protein=6, sugar=0, fats=3, white_meat=0, red_meat=0, fish=4, oil=2, milk=0, salts=0, caffeine=0),
        "fish with maalub":
            macros(protein=6, sugar=0, fats=4, white_meat=0, red_meat=0, fish=4, oil=3, milk=0, salts=0, caffeine=0),
        "fish and chips":
            macros(protein=5, sugar=2, fats=5, white_meat=0, red_meat=0, fish=4, oil=4, milk=0, salts=2, caffeine=0),
        "fish and eggs":
            macros(protein=7, sugar=1, fats=4, white_meat=0, red_meat=0, fish=4, oil=4, milk=0, salts=2, caffeine=0),
        "fish snd pasta":
            macros(protein=6, sugar=3, fats=4, white_meat=0, red_meat=0, fish=7, oil=4, milk=0, salts=0, caffeine=0),
        "zaatar":
            macros(protein=0, sugar=1, fats=0, white_meat=0, red_meat=0, fish=0, oil=1, milk=0, salts=2, caffeine=0),
        "cake":
            macros(protein=0, sugar=4, fats=1, white_meat=0, red_meat=0, fish=0, oil=0, milk=2, salts=0, caffeine=0),
        "cake small":
            macros(protein=0, sugar=2, fats=1, white_meat=0, red_meat=0, fish=0, oil=0, milk=1, salts=0, caffeine=0),
        "soup":
            macros(protein=0, sugar=0, fats=0, white_meat=0, red_meat=0, fish=0, oil=0, milk=0, salts=1, caffeine=0),
        "maalub":
            macros(protein=0, sugar=1, fats=1, white_meat=0, red_meat=0, fish=0, oil=0, milk=0, salts=1, caffeine=0),
        "mukpatz":
            macros(protein=1, sugar=2, fats=1, white_meat=2, red_meat=2, fish=0, oil=2, milk=0, salts=0, caffeine=0),
        "rice and ktzitzot":
            macros(protein=1, sugar=2, fats=1, white_meat=0, red_meat=2, fish=0, oil=2, milk=0, salts=0, caffeine=0),
        "schnitzel in a baguette":
            macros(protein=2, sugar=3, fats=0, white_meat=2, red_meat=0, fish=0, oil=0, milk=0, salts=0, caffeine=0),
        "mom's spaghetti":
            macros(protein=2, sugar=2, fats=1, white_meat=2, red_meat=0, fish=0, oil=2, milk=0, salts=0, caffeine=0),
        "kariot":
            macros(protein=0, sugar=4, fats=0, white_meat=0, red_meat=0, fish=0, oil=0, milk=2, salts=0, caffeine=0),
        "kariot small":
            macros(protein=0, sugar=2, fats=0, white_meat=0, red_meat=0, fish=0, oil=0, milk=1, salts=0, caffeine=0),
        "kariot large":
            macros(protein=0, sugar=6, fats=0, white_meat=0, red_meat=0, fish=0, oil=0, milk=3, salts=0, caffeine=0),
        "rice and fish":
            macros(protein=3, sugar=0, fats=1, white_meat=0, red_meat=0, fish=3, oil=1, milk=0, salts=0, caffeine=0),
        "rice with chicken":
            macros(protein=3, sugar=0, fats=0, white_meat=3, red_meat=0, fish=0, oil=0, milk=0, salts=0, caffeine=0),
        "fritata sandwich":
            macros(protein=3, sugar=2, fats=0, white_meat=0, red_meat=0, fish=0, oil=1, milk=0, salts=1, caffeine=0),
        "rice and schnitzel":
            macros(protein=3, sugar=0, fats=0, white_meat=3, red_meat=0, fish=0, oil=1, milk=0, salts=0, caffeine=0),
        "susi":
            macros(protein=4, sugar=0, fats=2, white_meat=0, red_meat=0, fish=4, oil=0, milk=0, salts=0, caffeine=0),
        "susi small":
            macros(protein=2, sugar=0, fats=1, white_meat=0, red_meat=0, fish=2, oil=0, milk=0, salts=0, caffeine=0),
        "beer":
            macros(protein=1, sugar=1, fats=0, white_meat=0, red_meat=0, fish=0, oil=0, milk=0, salts=3, caffeine=0),
        "dganim":
            macros(protein=2, sugar=3, fats=0, white_meat=0, red_meat=0, fish=0, oil=0, milk=0, salts=0, caffeine=0),
        "dganim small":
            macros(protein=1, sugar=1, fats=0, white_meat=0, red_meat=0, fish=0, oil=0, milk=0, salts=0, caffeine=0),
        "dganim large":
            macros(protein=3, sugar=5, fats=0, white_meat=0, red_meat=0, fish=0, oil=0, milk=0, salts=0, caffeine=0),
        "schnitzel with mugdara":
            macros(protein=3, sugar=0, fats=1, white_meat=0, red_meat=0, fish=0, oil=1, milk=0, salts=0, caffeine=0),
        "sulties":
            macros(protein=0, sugar=0, fats=1, white_meat=0, red_meat=0, fish=0, oil=0, milk=0, salts=4, caffeine=0),
        "ice cream":
            macros(protein=1, sugar=4, fats=1, white_meat=0, red_meat=0, fish=0, oil=0, milk=4, salts=0, caffeine=0),
        "kuskus and chicken and eggs":
            macros(protein=5, sugar=0, fats=1, white_meat=3, red_meat=0, fish=0, oil=2, milk=0, salts=0, caffeine=0),
        "":
            macros(protein=0, sugar=0, fats=0, white_meat=0, red_meat=0, fish=0, oil=0, milk=0, salts=0, caffeine=0),
        "":
            macros(protein=0, sugar=0, fats=0, white_meat=0, red_meat=0, fish=0, oil=0, milk=0, salts=0, caffeine=0),
        "":
            macros(protein=0, sugar=0, fats=0, white_meat=0, red_meat=0, fish=0, oil=0, milk=0, salts=0, caffeine=0),
        "":
            macros(protein=0, sugar=0, fats=0, white_meat=0, red_meat=0, fish=0, oil=0, milk=0, salts=0, caffeine=0),
        "":
            macros(protein=0, sugar=0, fats=0, white_meat=0, red_meat=0, fish=0, oil=0, milk=0, salts=0, caffeine=0),
        "":
            macros(protein=0, sugar=0, fats=0, white_meat=0, red_meat=0, fish=0, oil=0, milk=0, salts=0, caffeine=0),
        "":
            macros(protein=0, sugar=0, fats=0, white_meat=0, red_meat=0, fish=0, oil=0, milk=0, salts=0, caffeine=0),
        "":
            macros(protein=0, sugar=0, fats=0, white_meat=0, red_meat=0, fish=0, oil=0, milk=0, salts=0, caffeine=0),
        "":
            macros(protein=0, sugar=0, fats=0, white_meat=0, red_meat=0, fish=0, oil=0, milk=0, salts=0, caffeine=0),
        "":
            macros(protein=0, sugar=0, fats=0, white_meat=0, red_meat=0, fish=0, oil=0, milk=0, salts=0, caffeine=0),
        "":
            macros(protein=0, sugar=0, fats=0, white_meat=0, red_meat=0, fish=0, oil=0, milk=0, salts=0, caffeine=0),
        "":
            macros(protein=0, sugar=0, fats=0, white_meat=0, red_meat=0, fish=0, oil=0, milk=0, salts=0, caffeine=0),
        "":
            macros(protein=0, sugar=0, fats=0, white_meat=0, red_meat=0, fish=0, oil=0, milk=0, salts=0, caffeine=0),
        "":
            macros(protein=0, sugar=0, fats=0, white_meat=0, red_meat=0, fish=0, oil=0, milk=0, salts=0, caffeine=0),
        "":
            macros(protein=0, sugar=0, fats=0, white_meat=0, red_meat=0, fish=0, oil=0, milk=0, salts=0, caffeine=0),
        "":
            macros(protein=0, sugar=0, fats=0, white_meat=0, red_meat=0, fish=0, oil=0, milk=0, salts=0, caffeine=0),
        "":
            macros(protein=0, sugar=0, fats=0, white_meat=0, red_meat=0, fish=0, oil=0, milk=0, salts=0, caffeine=0),
        "":
            macros(protein=0, sugar=0, fats=0, white_meat=0, red_meat=0, fish=0, oil=0, milk=0, salts=0, caffeine=0),
        "":
            macros(protein=0, sugar=0, fats=0, white_meat=0, red_meat=0, fish=0, oil=0, milk=0, salts=0, caffeine=0),
        "":
            macros(protein=0, sugar=0, fats=0, white_meat=0, red_meat=0, fish=0, oil=0, milk=0, salts=0, caffeine=0),
        "":
            macros(protein=0, sugar=0, fats=0, white_meat=0, red_meat=0, fish=0, oil=0, milk=0, salts=0, caffeine=0),
        "":
            macros(protein=0, sugar=0, fats=0, white_meat=0, red_meat=0, fish=0, oil=0, milk=0, salts=0, caffeine=0),
        "":
            macros(protein=0, sugar=0, fats=0, white_meat=0, red_meat=0, fish=0, oil=0, milk=0, salts=0, caffeine=0),
        "":
            macros(protein=0, sugar=0, fats=0, white_meat=0, red_meat=0, fish=0, oil=0, milk=0, salts=0, caffeine=0),
        "":
            macros(protein=0, sugar=0, fats=0, white_meat=0, red_meat=0, fish=0, oil=0, milk=0, salts=0, caffeine=0),
        "":
            macros(protein=0, sugar=0, fats=0, white_meat=0, red_meat=0, fish=0, oil=0, milk=0, salts=0, caffeine=0),
        "":
            macros(protein=0, sugar=0, fats=0, white_meat=0, red_meat=0, fish=0, oil=0, milk=0, salts=0, caffeine=0),

    }#     Protein     Sugar     Fats     White meat     Red meat     Fish     Milk     Caffeine

    if key == 4:
        return pee_dict.get(string, string)
    else:
        return shit_dict.get(string, string)


if __name__ == "__main__":
    main()
