import subprocess


folder = r"" + "\\"
video_in = folder + ".mp4"
audio_in = folder + ".wav"
video_out = folder + ".mp4"

command = ["ffmpeg",
           "-i", video_in,  # Input video
           "-i", audio_in,  # Input new audio
           "-c:v", "copy",    # Copy video codec
           "-c:a", "aac",     # Encode audio as AAC (MP4-compatible)
           "-strict", "experimental",
           "-map", "0:v:0",   # Use first video stream from input video
           "-map", "1:a:0",   # Use first audio stream from input audio
           "-y",              # Overwrite output file if exists
           video_out
]

subprocess.run(command, check=True)
print("Audio track replaced successfully!")
