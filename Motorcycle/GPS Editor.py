# pyinstaller --noconfirm --onedir --windowed --contents-directory "GPS Editor" --icon "GPS Editor.png"  "GPS Editor.py"


import os
import sys
import gpxpy
import folium
import webbrowser


FILE = r""
GPX_FILE_IN = FILE + ".gpx"
HTML_FILE_OUT = FILE + ".html"


# ## ### Enable for PyInstaller ### ## #

# if len(sys.argv) < 2:
#     print("Drag a GPX file onto this EXE")
#     sys.exit(1)
#
# GPX_FILE_IN = sys.argv[1]
# HTML_FILE_OUT = GPX_FILE_IN.replace(".gpx", ".html")

# ## ### Enable for PyInstaller ### ## #


with open(GPX_FILE_IN, "r", encoding="utf-8") as gpx_file:
    gpx = gpxpy.parse(gpx_file)

points = []
for track in gpx.tracks:
    for segment in track.segments:
        for point in segment.points:
            points.append((point.latitude, point.longitude))
if not points:
    raise Exception("No GPS points found")


start_lat, start_lon = points[0]
m = folium.Map(location=[start_lat, start_lon], zoom_start=14, tiles="https://{s}.tile.openstreetmap.de/{z}/{x}/{y}.png",
               attr='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors')
folium.PolyLine(points, color="blue", weight=4, opacity=0.8).add_to(m)                                                  # Track line
folium.Marker(points[0], popup="Start", icon=folium.Icon(icon="play", prefix="fa", color="green")).add_to(m)            # Start marker
folium.Marker(points[-1], popup="End", icon=folium.Icon(icon="flag-checkered", prefix="fa", color="red")).add_to(m)     # End marker

m.save(HTML_FILE_OUT)
webbrowser.open("file://" + os.path.realpath(HTML_FILE_OUT))
print(f"Map saved to {HTML_FILE_OUT}")
