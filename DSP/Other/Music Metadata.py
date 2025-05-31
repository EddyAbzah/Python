import os
from mutagen.mp3 import MP3
from mutagen.id3 import ID3, APIC, USLT, error


def find_cover_image(directory, artist, album):
    pattern = f"{artist} - {album}".lower()
    for file in os.listdir(directory):
        if file.lower().startswith(pattern) and file.lower().endswith((".png", ".jpg", ".jpeg")):
            return os.path.join(directory, file)
    return None


def process_mp3_files(directory):
    for file in os.listdir(directory):
        if not file.lower().endswith(".mp3"):
            continue

        mp3_path = os.path.join(directory, file)
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
            lyrics_path = os.path.join(directory, base_name + ext)
            if os.path.isfile(lyrics_path):
                with open(lyrics_path, "r", encoding="utf-8", errors="ignore") as f:
                    lyrics = f.read()
                id3.setall("USLT", [USLT(encoding=3, lang='eng', desc='', text=lyrics)])
                added_lyrics = True
                lyric_ext = ext
                break  # Only use the first found

        # === Embed album art ===
        artist = id3.get("TPE1")
        album = id3.get("TALB")
        if artist and album:
            artist_str = artist.text[0]
            album_str = album.text[0]
            image_path = find_cover_image(directory, artist_str, album_str)
            if image_path:
                with open(image_path, 'rb') as img:
                    id3.setall("APIC", [APIC(
                        encoding=3,
                        mime='image/jpeg' if image_path.endswith(".jpg") else 'image/png',
                        type=3,
                        desc='Cover',
                        data=img.read()
                    )])
                added_cover = True

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
