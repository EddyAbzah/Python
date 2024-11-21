import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d

# Specify the input and output file paths
file_in = r""
file_out = r""
df_round = 2
noise_level = 0.005  # 1 = 100%
noise_level = 0


class InteractivePlot:
    def __init__(self, ax, x_data, noise_level=0.0):
        self.ax = ax
        self.fig = ax.figure
        self.x_data = x_data  # x-values from the CSV data
        self.noise_level = noise_level  # Noise level to add to the data
        # Initialize an empty line
        self.line, = ax.plot([], [], 'r-', label='Your Drawing')
        # Lists to store x and y data points
        self.xs = []
        self.ys = []
        self.is_pressed = False
        self.user_y_values = None  # To store interpolated y-values at x_data

        # Connect the event handlers
        self.cid_press = self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.cid_release = self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.cid_motion = self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)

    def on_press(self, event):
        """Handle mouse button press events."""
        if event.inaxes != self.ax:
            return
        self.is_pressed = True
        self.xs = [event.xdata]
        self.ys = [event.ydata]
        self.line.set_data(self.xs, self.ys)
        self.fig.canvas.draw()

    def on_release(self, event):
        """Handle mouse button release events."""
        if not self.is_pressed:
            return
        self.is_pressed = False
        # Interpolate the drawn line to the x_data points
        if len(self.xs) >= 2:
            # Sort the drawn data by x-values
            sorted_indices = np.argsort(self.xs)
            xs_sorted = np.array(self.xs)[sorted_indices]
            ys_sorted = np.array(self.ys)[sorted_indices]
            # Remove duplicate x-values
            xs_unique, indices = np.unique(xs_sorted, return_index=True)
            ys_unique = ys_sorted[indices]
            # Create interpolation function
            interp_func = interp1d(
                xs_unique, ys_unique, kind='linear', bounds_error=False, fill_value="extrapolate"
            )
            # Get y-values at the x_data points
            self.user_y_values = interp_func(self.x_data)
            # Add noise to the user-drawn data
            if self.noise_level > 0:
                max_abs_y = np.max(np.abs(self.user_y_values))
                noise = np.random.normal(
                    0, self.noise_level * max_abs_y, size=self.user_y_values.shape
                )
                self.user_y_values += noise
            # Update the line data to the interpolated values
            self.line.set_data(self.x_data, self.user_y_values)
            self.fig.canvas.draw()
        else:
            print("Please draw a complete line with at least two points.")

    def on_motion(self, event):
        """Handle mouse movement events."""
        if not self.is_pressed or event.inaxes != self.ax:
            return
        self.xs.append(event.xdata)
        self.ys.append(event.ydata)
        self.line.set_data(self.xs, self.ys)
        self.fig.canvas.draw()


if file_in == "":
    file_in = input("Enter the input file:\n")
    file_in = file_in.replace('"', '')
if file_out == "":
    file_out = input("Enter the output file:\n")
    file_out = file_out.replace('"', '')

# Load data from CSV using the first column as the index
df = pd.read_csv(file_in, index_col=0)
x_data = df.index.values  # x_data from the index

# Display available traces to the user
print("Available traces:")
for i, col in enumerate(df.columns):
    print(f"{i}: {col}")

# Prompt the user to select traces
selected_input = input("Enter the indices of the traces you want to select (comma-separated), or 'all' to select all traces:\n")
if selected_input.strip().lower()[:1] == 'a':
    selected_indices = list(range(len(df.columns)))
else:
    selected_indices = [int(idx.strip()) for idx in selected_input.split(',')]

# Copy the original DataFrame to include unedited traces
user_df = df.copy()
user_df.index.name = df.index.name

for idx in selected_indices:
    col_name = df.columns[idx]
    y_data = df[col_name].values

    # Create a figure and axis
    fig, ax = plt.subplots()
    ax.set_title(f'Draw Over the Data for {col_name}')
    ax.plot(x_data, y_data, 'b-', label=f'Original Data - {col_name}')

    # Optionally, set axis labels
    ax.set_xlabel(user_df.index.name)
    ax.set_ylabel(col_name)

    # Initialize the interactive plot with x_data from CSV and noise_level
    interactive_plot = InteractivePlot(ax, x_data, noise_level=noise_level)

    # Display the plot with a legend
    plt.legend()
    plt.show()

    # After the plot window is closed, access the user's y-values at x_data points
    user_y_values = interactive_plot.user_y_values
    if user_y_values is not None:
        # Add the user-drawn y-values to the user_df DataFrame
        user_df[col_name] = user_y_values
        print(f'User-drawn data for "{col_name}" captured.')
    else:
        print(f'No data was drawn for "{col_name}".')


# Apply rounding if specified
if df_round == 0:
    user_df = user_df.apply(int)
elif df_round > 0:
    user_df = user_df.round(df_round)

# Save the user-drawn data to the specified output file, including the index and its name
user_df.to_csv(file_out, index=True)
print(f"User-drawn data saved to {file_out}")
