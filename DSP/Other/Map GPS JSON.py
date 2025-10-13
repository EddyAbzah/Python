import os
import json
import folium
import webbrowser
from datetime import datetime


file_in = r""
file_out = r""


def parse_latlng(latlng_str):
    """Convert string like '33.0484691°, 35.4883511°' into (lat, lon) floats."""
    try:
        lat_str, lon_str = latlng_str.replace("°", "").split(",")
        return float(lat_str.strip()), float(lon_str.strip())
    except Exception:
        return None


def format_time(iso_str):
    """Convert ISO string like '2025-10-10T11:20:00.000+03:00' to '2025-10-10 _ 11-20-00'."""
    try:
        # Replace T and timezone for parsing
        clean = iso_str.split("+")[0].replace("T", " ")
        dt = datetime.fromisoformat(clean)
        return dt.strftime("%Y-%m-%d _ %H-%M-%S")
    except Exception:
        return iso_str  # return unchanged if parsing fails


def get_gps(index, segment):
    """
    Extract GPS coordinates and readable labels from a Google semantic segment.
    Returns a list of ((lat, lon), label).
    """
    coords = []
    print(f'{index:03} = {segment}')

    # 1. Timeline paths
    if "timelinePath" in segment:
        for point in segment["timelinePath"]:
            if "point" in point:
                gps = parse_latlng(point["point"])
                if gps:
                    label = format_time(point.get("time", "Unknown time"))
                    label = f'{index:03} _ {label}'
                    coords.append((gps, label))

    # 2. Activity start / end
    if "activity" in segment:
        act = segment["activity"]
        for key in ("start", "end"):
            loc = act.get(key, {}).get("latLng")
            if loc:
                gps = parse_latlng(loc)
                if gps:
                    start_t = segment.get("startTime", "")
                    label = f"{key.title()} ({format_time(start_t)})"
                    label = f'{index:03} _ {label}'
                    coords.append((gps, label))

        # 3. Parking
        parking = act.get("parking", {}).get("location", {}).get("latLng")
        if parking:
            gps = parse_latlng(parking)
            if gps:
                t = act.get("parking", {}).get("startTime", "Parking")
                label = f"Parking ({format_time(t)})"
                label = f'{index:03} _ {label}'
                coords.append((gps, label))

    # 4. Visits
    if "visit" in segment:
        loc = segment["visit"].get("topCandidate", {}).get("placeLocation", {}).get("latLng")
        if loc:
            gps = parse_latlng(loc)
            if gps:
                start_t = segment.get("startTime", "")
                label = f"Visit ({format_time(start_t)})"
                label = f'{index:03} _ {label}'
                coords.append((gps, label))

    return coords


# Load JSON
with open(file_in, "r", encoding="utf-8") as f:
    data = json.load(f)

segments = data.get("semanticSegments", [])
coordinates = []

for i, seg in enumerate(segments, start=1):
    gps_points = get_gps(i, seg)
    coordinates.extend(gps_points)

if not coordinates:
    print("No coordinates found in JSON file.")
    exit()

# Create map
m = folium.Map(location=coordinates[0][0], zoom_start=10, tiles="OpenStreetMap")

# Draw blue path
folium.PolyLine([coord for coord, _ in coordinates], color="blue", weight=2.5).add_to(m)

# Add markers every N points
N = max(len(coordinates) // 100, 1)
for coord, label in coordinates[::N]:
    folium.Marker(coord, popup=label).add_to(m)

# Save and open
m.save(file_out)
print(f"✅ Map saved to: {file_out}")

webbrowser.open('file://' + os.path.abspath(file_out))
