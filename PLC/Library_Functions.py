import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def print_chrome(df, path, file_name, auto_open=True):
    fig = make_subplots(cols=1, rows=2, shared_xaxes=False, subplot_titles=("Linear Scale", "Logarithmic Scale"))
    for r in range(1, 3):
        for index, (info, trace) in enumerate(df.items()):
            fig.add_trace(go.Scatter(x=list(df.index), y=trace, name=info), col=1, row=r)
    fig.update_layout(title=file_name, title_font_color="#407294", title_font_size=40, legend_title="Plots:")
    fig.update_xaxes(title_text="Frequency [Hz] - Linear scale", row=1, col=1)
    fig.update_xaxes(title_text="Frequency [Hz] - Logarithmic scale", type="log", row=2, col=1)
    plotly.offline.plot(fig, config={'scrollZoom': True, 'editable': True}, filename=f'{path}\\{file_name}.html', auto_open=auto_open)
