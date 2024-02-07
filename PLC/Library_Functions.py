import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def print_chrome(df, path, file_name, scale='both', auto_open=True):
    """
    Args:
        scale: String. x-axis scale. set to either 'linear', 'logarithmic' or 'both'; default = 'both'.
    """
    subplots = []
    if 'log' not in scale:
        subplots.append("Linear Scale")
    if 'lin' not in scale:
        subplots.append("Logarithmic Scale")
    fig = make_subplots(cols=1, rows=len(subplots), shared_xaxes=False, subplot_titles=subplots)
    for i, r in enumerate(subplots):
        for index, (info, trace) in enumerate(df.items()):
            fig.add_trace(go.Scatter(x=list(df.index), y=trace, name=info), col=1, row=i + 1)
        fig.update_xaxes(title_text="Frequency [Hz] - " + r, col=1, row=i + 1, type='log' if r == "Logarithmic Scale" else None)
    fig.update_layout(title=file_name, title_font_color="#407294", title_font_size=40, legend_title="Plots:")
    plotly.offline.plot(fig, config={'scrollZoom': True, 'editable': True}, filename=f'{path}\\{file_name}.html', auto_open=auto_open)
