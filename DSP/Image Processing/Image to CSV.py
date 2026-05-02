import matplotlib.pyplot as plt
import pandas as pd
import cv2
import numpy as np


file_in = r".jpg"
file_out = r".csv"
x_axis = [1000, 10500, 25]
y_axis = [0, 80, 2]
plot_interpolated_curve = True


def on_press(event):
    global dragging
    dragging = True


def on_release(event):
    global dragging
    dragging = False


def on_motion(event):
    if dragging and event.xdata is not None and event.ydata is not None:
        x, y = event.xdata, event.ydata
        points.append((x, y))
        plt.plot(x, y, 'r.', markersize=2)
        plt.draw()


img = cv2.imread(file_in)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
points = []
dragging = False

fig, ax = plt.subplots()
ax.imshow(img)
ax.set_title("Click and drag along the curve, then close window when done")

fig.canvas.mpl_connect('button_press_event', on_press)
fig.canvas.mpl_connect('button_release_event', on_release)
fig.canvas.mpl_connect('motion_notify_event', on_motion)

plt.show()

points = np.array(points)
if len(points) == 0:
    print("No points traced. Exiting.")
    exit()

points[:, 1] = img.shape[0] - points[:, 1]
points = points[points[:, 0].argsort()]

x_min_pixel = points[0, 0]
x_max_pixel = points[-1, 0]
x_values_real = x_axis[0] + (points[:, 0] - x_min_pixel) / (x_max_pixel - x_min_pixel) * (x_axis[1] - x_axis[0])

y_min_pixel = min(points[:, 1])
y_max_pixel = max(points[:, 1])
y_values_real = y_axis[0] + (points[:, 1] - y_min_pixel) / (y_max_pixel - y_min_pixel) * (y_axis[1] - y_axis[0])

x_interp = np.arange(x_axis[0], x_axis[1]+x_axis[2], x_axis[2])
y_interp = np.interp(x_interp, x_values_real, y_values_real)
y_interp = np.round(y_interp, y_axis[2])

df = pd.DataFrame({"X": x_interp, "Y": y_interp})
df.to_csv(file_out, index=False)

print(f"CSV saved to {file_out}, points interpolated: {len(x_interp)}")

if plot_interpolated_curve:
    plt.plot(x_values_real, y_values_real, 'r.', label='Traced Points')
    plt.plot(x_interp, y_interp, 'b-', label='Interpolated Curve')
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.legend()
    plt.show()
