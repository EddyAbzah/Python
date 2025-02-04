import os
import subprocess


program_path = r"C:\Users\eddya\OneDrive\מסמכים\MPV Player\mpv.exe"
playlist_file_path = r"C:\Users\eddya\Videos\MPV Player Playlist.txt"
directories = [r"",
               r""]
select_file = 1


files = [f for f in os.listdir(directories[0]) if f.lower().endswith('.mp4')]
playlist = [os.path.join(d, files[select_file - 1]) + "\n" for d in directories]

with open(playlist_file_path, "w") as file:
    file.writelines(playlist)
print(*playlist, sep="", end="")

subprocess.Popen([program_path, f"--playlist={playlist_file_path}", "--load-unsafe-playlists"])
