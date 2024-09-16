import os
import moviepy.editor as mpy


video_type_select = 0
video_codec = ["libx264", "libtheora", "libvpx", "libvorbis", "pcm_s16le", "libvorbis", "libfdk_aac"][video_type_select]
video_file_extension = [".mp4", ".ogv", ".webm", ".ogg", ".mp3", ".wav", ".m4a"][video_type_select]
video_quality = "24"
compression = ["slow", "ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow", "slower", "veryslow"][0]
fps = -30

file_path = r"C:\Users\eddy.a\Downloads" + "\\"
file_in = 'rapidsave.com_bro-o8ezsnlpexyc1.mp4'
file_out = file_in.split('.')[0] + ' (edit)' + video_file_extension

# modify these start and end times for your subclips
subclips = [['00:00:01.000', '']]
add_text = ''

# load file
video = mpy.VideoFileClip(file_path + file_in)
if fps <= 0:
    fps = video.fps

# cut file
clips = []
for cut in subclips:
    if cut[0] == '':
        cut[0] = video.start
    if cut[1] == '':
        cut[1] = video.end
    clip = video.subclip(cut[0], cut[1])
    clips.append(clip)
final_clip = mpy.concatenate_videoclips(clips)

# add text
if add_text != '':
    txt = mpy.TextClip(add_text, font='Courier', fontsize=120, color='white', bg_color='gray35')
    txt = txt.set_position(('center', 0.6), relative=True)
    txt = txt.set_start((0, 3))  # (min, s)
    txt = txt.set_duration(4)
    txt = txt.crossfadein(0.5)
    txt = txt.crossfadeout(0.5)
    final_clip = mpy.CompositeVideoClip([final_clip, txt])

# save file
final_clip.write_videofile(file_path + file_out, threads=4, fps=30, codec=video_codec, preset=compression, ffmpeg_params=["-crf", video_quality])
video.close()
