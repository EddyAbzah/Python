import os, re
from mutagen.mp3 import MP3
from mutagen.id3 import ID3, APIC, USLT, error


DELETE_SOURCE_FILES = True  # Set to True to delete embedded files
CLEAN_LYRICS = True         # Set to True to remove lines with [hh:mm.ss] format
used_lyric_files = set()
used_image_files = set()


def find_cover_image(directory, artist, album):
    pattern = f"{artist} - {album}".lower()
    for file in os.listdir(directory):
        if file.lower().startswith(pattern) and file.lower().endswith((".png", ".jpg", ".jpeg")):
            return os.path.join(directory, file)
    return None


def clean_lyrics_text(text):
    return "\n".join(line for line in text.splitlines() if not re.match(r"^\[\d{2}:\d{2}\.\d{2,3}\]", line.strip()))


def process_mp3_files(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            if not file.lower().endswith(".mp3"):
                continue

            mp3_path = os.path.join(root, file)
            base_name = os.path.splitext(file)[0]
            added_lyrics = False
            added_cover = False
            lyric_ext = None

            try:
                audio = MP3(mp3_path, ID3=ID3)
                id3 = audio.tags
            except error:
                print(f"{mp3_path}: file not found")
                continue

            # === Add lyrics if matching .txt or .lrc exists ===
            for ext in [".txt", ".lrc"]:
                lyrics_path = os.path.join(root, base_name + ext)
                if os.path.isfile(lyrics_path):
                    with open(lyrics_path, "r", encoding="utf-8", errors="ignore") as f:
                        lyrics = f.read()
                    if CLEAN_LYRICS:
                        lyrics = clean_lyrics_text(lyrics)
                    id3.setall("USLT", [USLT(encoding=3, lang='eng', desc='', text=lyrics)])
                    added_lyrics = True
                    lyric_ext = ext
                    if DELETE_SOURCE_FILES:
                        used_lyric_files.add(lyrics_path)
                        print(f'{lyrics_path = }')
                    break  # Only use the first found

            # === Embed album art ===
            artist = id3.get("TPE1")
            album = id3.get("TALB")
            if artist and album:
                artist_str = artist.text[0]
                album_str = album.text[0]
                image_path = find_cover_image(root, artist_str, album_str)
                if image_path:
                    with open(image_path, 'rb') as img:
                        id3.setall("APIC", [APIC(encoding=3, mime='image/jpeg' if image_path.endswith(".jpg") else 'image/png', type=3, desc='Cover', data=img.read())])
                    added_cover = True
                    if DELETE_SOURCE_FILES:
                        used_image_files.add(image_path)
                        print(f'{image_path = }')
            id3.save(mp3_path, v2_version=4)

            # === Final output ===
            if added_lyrics and added_cover:
                print(f"{mp3_path}: add {lyric_ext} lyric file and album cover")
            elif added_lyrics:
                print(f"{mp3_path}: add {lyric_ext} lyric file")
            elif added_cover:
                print(f"{mp3_path}: add album cover")
            else:
                print(f"{mp3_path}: file not found")


if __name__ == '__main__':
    directory = r""
    process_mp3_files(directory)
    if DELETE_SOURCE_FILES:
        for lyric_file in used_lyric_files:
            if os.path.isfile(lyric_file):
                os.remove(lyric_file)
        for image_file in used_image_files:
            if os.path.isfile(image_file):
                os.remove(image_file)
