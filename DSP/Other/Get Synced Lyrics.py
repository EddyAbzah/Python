"""
https://github.com/moehmeni/syncedlyrics

By default, this will prefer time synced lyrics, but use plaintext lyrics, if no synced lyrics are available. To only allow one type of lyrics specify --plain-only or --synced-only respectively.
Available Options
Flag 	Description
-o 	Path to save .lrc lyrics, default="{search_term}.lrc"
-p 	Space-separated list of providers to include in searching
-l 	Language code of the translation (ISO 639-1 format)
-v 	Use this flag to show the logs
--plain-only 	Only look for plain text (not synced) lyrics
--synced-only 	Only look for synced lyrics
--enhanced 	Searches for an Enhanced (word-level karaoke) format. If it isn't available, search for regular synced lyrics.

Providers:
 - Musixmatch
 - Lrclib
 - NetEase
 - Megalobiz
 - Genius (For plain format)
"""

import os
import re
import syncedlyrics


def download_lyrics(full_path, root, file):
    synced_only = False
    # providers = ["Musixmatch", "Lrclib", "NetEase", "Megalobiz", "Genius"]
    # lyrics = syncedlyrics.search(f"[{file}] [{root}]", synced_only=synced_only, providers=providers)
    lyrics = syncedlyrics.search(f"[{file}] [{root}]", synced_only=synced_only)
    if lyrics:
        lyrics = [line for line in lyrics.split('\n') if not re.search(r'[\u4e00-\u9fff]', line) and 'contributor' not in line.lower() and 'lyric' not in line.lower()]

        if re.search(r"\[\d{1,2}:\d{2}\.\d{1,3}\]", lyrics[0]):
            extension = ".lrc"
        else:
            extension = ".txt"
        full_path = full_path[:-4] + extension

        with open(full_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(lyrics))
            print(f'{root = }, {file = }, output = {full_path}')
    else:
        print(f'{root = }, {file = }, NO LYRICS')


base_directory = r"C:\Users\eddya\Music"
filter_out = ["_MusicBee", "_Practice Guitars", "Various Adyghe", "Various Classical", "Various Languages", "Various Podcasts", "Various Soundtrack"]
filter_special = ["Classic", "Various"]

for root, dirs, files in os.walk(base_directory):
    if any(fo.lower() in root.lower() for fo in filter_out):
        continue
    for file in files:
        if file.lower().endswith('.mp3'):
            full_path = os.path.join(root, file)
            if not any(os.path.exists(full_path[:-4] + ext) for ext in ('.txt', '.lrc', '.lyric')):
                if any(fs.lower() in root.lower() for fs in filter_special):
                    download_lyrics(full_path, file.split(" - ")[0], file.split(" - ")[1][:-4])
                else:
                    file = file[10:-4]
                    download_lyrics(full_path, os.path.basename(root), file)
