import os
import re
import folium
import exifread
from pymediainfo import MediaInfo


def get_gps_from_image(img_path):
    """Extract GPS coordinates from image EXIF data."""
    with open(img_path, 'rb') as f:
        tags = exifread.process_file(f, stop_tag="GPS GPSLongitude")

    def get_decimal(coord, ref):
        degrees = coord[0].num / coord[0].den
        minutes = coord[1].num / coord[1].den
        seconds = coord[2].num / coord[2].den
        decimal = degrees + (minutes / 60.0) + (seconds / 3600.0)
        if ref in ['S', 'W']:
            decimal = -decimal
        return decimal

    try:
        lat_tag = tags['GPS GPSLatitude']
        lat_ref = tags['GPS GPSLatitudeRef'].printable
        lon_tag = tags['GPS GPSLongitude']
        lon_ref = tags['GPS GPSLongitudeRef'].printable

        lat = get_decimal(lat_tag.values, lat_ref)
        lon = get_decimal(lon_tag.values, lon_ref)
        return lat, lon
    except KeyError:
        return None


def get_gps_from_video(video_path):
    """Extract GPS coordinates from video metadata (if available)."""
    try:
        media_info = MediaInfo.parse(video_path)
        for track in media_info.tracks:
            if track.track_type == "Video" or track.track_type == "General":
                for key, value in track.to_data().items():
                    if "xyz" in key:
                        match = re.match(r'^([+-]\d+\.\d+)([+-]\d+\.\d+)/?$', value)
                        if match:
                            lat = float(match.group(1))
                            lon = float(match.group(2))
                            return lat, lon
    except:
        return None


def build_map(folder, subfolder):
    """Extract GPS data from images/videos and build a map."""
    coordinates = []
    folder_path = folder + "\\" + subfolder

    for filename in os.listdir(folder_path):
        full_path = os.path.join(folder_path, filename)
        lower = filename.lower()
        gps = None

        if lower.endswith(('.jpg', '.jpeg')):
            gps = get_gps_from_image(full_path)
        elif lower.endswith(('.mp4', '.mov', '.avi', '.mkv')):
            gps = get_gps_from_video(full_path)
        print(f'{filename = }\t→\t{gps = }')
        if gps:
            coordinates.append(gps)

    if not coordinates:
        print("No GPS data found in images or videos.")
        return

    # Create map with English tile layer
    m = folium.Map(location=coordinates[0], zoom_start=13, tiles="OpenStreetMap")

    # Draw path
    folium.PolyLine(coordinates, color="blue", weight=2.5).add_to(m)

    # Add markers
    for coord in coordinates:
        folium.Marker(coord).add_to(m)

    output_file = os.path.join(folder, subfolder + ".html")
    m.save(output_file)
    print(f"Map saved to: {output_file}")


if __name__ == "__main__":
    folder = r""
    subfolder = ""
    build_map(folder, subfolder)
