"""
    Create and plot graphs with sliders in Chrome using plotly
    To use, enter line:
    import Plot_Graphs_with_Sliders as _G

    Example:
    _G.add_data(df_a, 'df_a title')
    _G.add_data(df_b, 'df_b title')
    _G.add_data(df_c, 'df_c title', 0)
    _G.set_slider([-1, 0, 1], ['all', 'slider = df_a', 'slider = df_b'])
    _G.plot()
    _G.reset()

    @author: Eddy Abzah
    @last update: 07/11/2022
"""


# ## ### #### ##### Imports ##### #### ### ## #
import os
from datetime import datetime
import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from my_pyplot import plot as _P, print_chrome as _PC, clear as _PP, print_lines as _PL, send_mail as _SM
import Plot_Graphs_with_Sliders as _G
import my_tools


# ## ### #### ##### Properties ##### #### ### ## #
path_output = r'M:\Users\HW Infrastructure\PLC team\ARC\Temp-Eddy'
kill_chrome = False
auto_open_chrome_output = False
plot_name = f'Over Power Vdc Arcs'
slider_prefix = 'Slider = '
shared_x_axis = True
# ## ### figs: col, row
plot_shape = {1: [1, 1], 2: [1, 2], 3: [1, 3], 4: [2, 2], 5: [1, 5], 6: [3, 2], 7: [1, 7], 8: [4, 2], 9: [3, 3], 10: [5, 2]}


# ## ### #### ##### Data ##### #### ### ## #
# ## [plot, title, index]
plots = []
plot_titles = []
plot_matrix = []
plot_counter = 0
plot_sliders = [False, [], []]


def add_data(df, title, index=0.1, true_if_matrix=False, title_matrix=''):
    global plots
    global plot_titles
    global plot_matrix
    global plot_counter
    if index != 0.1:
        if len(plots) < index:
            raise Exception(f'Can not use "plots" index = {index} because the len is = {len(plots)}')
        elif len(plots) == index:
            plots.append([df, title])
            plot_titles.append(title)
            plot_matrix.append(False)
            plot_counter += 1
        else:
            if plot_matrix[index] == False:
                plots[index] = [plots[index]]
            plots[index].append([df, title])
            if title_matrix == '':
                plot_titles[index] = f'{plot_titles[index]} + {title}'
            else:
                plot_titles[index] = title_matrix
            plot_matrix[index] = True
            plot_counter += 1
    elif true_if_matrix:
        plots.append([[a, b] for a, b in zip(df, title)])
        if title_matrix == '':
            plot_titles.append(' + '.join(title))
        else:
            plot_titles.append(title_matrix)
        plot_matrix.append(True)
        plot_counter += len(df)
    else:
        plots.append([df, title])
        plot_titles.append(title)
        plot_matrix.append(False)
        plot_counter += 1


def set_slider(indexes, sliders_titles, true_to_unset=False):
    global plot_sliders
    if true_to_unset:
        plot_sliders = [False, [], []]
    else:
        plot_sliders = [True, indexes, sliders_titles]


def plot():
    print(f"Plot Graphs with Sliders - starting plot(): Time = {str(datetime.now())[:-7]}")
    if not os.path.exists(path_output):
        os.makedirs(path_output)
    plot_columns = plot_shape[len(plot_titles)][0]
    plot_rows = plot_shape[len(plot_titles)][1]
    all_specs = np.reshape([[{"secondary_y": False}] for x in range(len(plot_titles))], (plot_rows, plot_columns)).tolist()
    fig = make_subplots(subplot_titles=plot_titles, rows=plot_rows, cols=plot_columns, specs=all_specs, shared_xaxes=shared_x_axis)
    for index_plot, plot_title in enumerate(plot_titles):
        for plt in plots[index_plot]:
            if not plot_matrix[index_plot]:
                plt = plots[index_plot]
            fig.add_trace(go.Scatter(y=plt[0], name=plt[1]), col=index_plot % plot_columns + 1, row=int(index_plot / plot_columns) + 1)
            if not plot_matrix[index_plot]:
                break

    if plot_sliders[0]:
        slider_steps = []
        all_steps = [True] * plot_counter
        for index, index_plot in enumerate(plot_sliders[1]):
            if type(index_plot) == list:
                for index_plot_2 in index_plot:
                    if index_plot_2 >= 0:
                        all_steps[index_plot_2] = False
                        if index != 0:
                            fig.data[index_plot_2].visible = False
            if type(index_plot) != list and index_plot >= 0:
                all_steps[index_plot] = False
                if index != 0:
                    fig.data[index_plot].visible = False
        for index, index_plot in enumerate(plot_sliders[1]):
            if type(index_plot) != list and index_plot < 0:
                all_steps_new = [True] * plot_counter
            else:
                all_steps_new = all_steps.copy()
                if type(index_plot) == list:
                    for index_plot_2 in index_plot:
                        all_steps_new[index_plot_2] = True
                else:
                    all_steps_new[index_plot] = True
            step = dict(args=[{"visible": all_steps_new}, ], label=plot_sliders[2][index])
            slider_steps.append(step)
        fig.update_layout(sliders=[dict(active=0, pad={"t": 50}, steps=slider_steps, bgcolor="#ffb200", currentvalue=dict(prefix=slider_prefix, xanchor="center", font=dict(size=16)))])
    fig.update_layout(title=plot_name, title_font_color="#407294", title_font_size=40, legend_title="Records:", legend_title_font_color="green")
    if kill_chrome:
        os.system("taskkill /im chrome.exe /f")
    plotly.offline.plot(fig, config={'scrollZoom': True, 'editable': True}, filename=f'{path_output}/{plot_name}.html', auto_open=auto_open_chrome_output)
    print(f"Plot Graphs with Sliders - finishing plot(): Time = {str(datetime.now())[:-7]}")


def reset():
    global plots
    global plot_titles
    global plot_counter
    global plot_matrix
    global plot_sliders
    plots = []
    plot_titles = []
    plot_matrix = []
    plot_counter = 0
    plot_sliders = [False, [], []]


def remove_plot(index):
    global plots
    global plot_titles
    global plot_counter
    global plot_matrix
    global plot_sliders
    del plots[index]
    del plot_titles[index]
    del plot_matrix[index]
    if plot_sliders[0] and index in plot_sliders[1]:
        plot_sliders = [False, [], []]


def change_path(new_path):
    global path_output
    path_output = new_path


def change_name(new_name):
    global plot_name
    plot_name = new_name


def change_action(new_kill_chrome=kill_chrome, new_auto_open_chrome_output=auto_open_chrome_output, new_shared_x_axis=shared_x_axis, new_slider_prefix=slider_prefix):
    global kill_chrome
    global auto_open_chrome_output
    global shared_x_axis
    global slider_prefix
    kill_chrome = new_kill_chrome
    auto_open_chrome_output = new_auto_open_chrome_output
    shared_x_axis = new_shared_x_axis
    slider_prefix = new_slider_prefix
