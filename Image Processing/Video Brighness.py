import os
import sys
import cv2
# import ffmpeg
import numpy as np
from tqdm import tqdm


T_brightness__F_normalize = True
brightness_average = []
smoothing_factor = 0
video_codec = ['H264', 'X264', 'mp4v'][2]
tqdm_color = 'cyan'


def adjust_brightness(frame):
    """Function to increase brightness by adding a constant to each pixel
    cv2.convertScaleAbs: alpha = contrast; beta = brightness"""
    brightness_adjustment = brightness_average - get_frame_brightness(frame)
    # Convert frame to a new type to prevent overflow, then add brightness
    result_frame = cv2.convertScaleAbs(frame, alpha=1, beta=brightness_adjustment)
    return result_frame


def get_frame_brightness(frame):
    """Calculate the average brightness of the frame by converting it to grayscale"""
    grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    avg_brightness = np.mean(grayscale_frame)
    return avg_brightness


def normalize_lighting(frame):
    """Function to normalize lighting using histogram equalization"""
    # Convert the frame to YUV (Y: brightness, U, V: color channels)
    yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
    # Equalize the histogram of the Y channel (brightness)
    yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
    # Convert back to BGR
    result_frame = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
    return result_frame


def smooth_lighting(frames):
    """Function to smooth lighting transitions between frames"""
    smoothed_frames = [frames[0]]  # Start with the first frame
    for i in tqdm(range(1, len(frames)), desc="Smoothing lighting transitions", file=sys.stdout, colour=tqdm_color):
        # Blend the previous frame with the current one
        blended_frame = cv2.addWeighted(smoothed_frames[i - 1], smoothing_factor, frames[i], 1 - smoothing_factor, 0)
        smoothed_frames.append(blended_frame)
    return smoothed_frames


# def merge_audio_with_ffmpeg_python(original_video_path, processed_video_path):
#     video = ffmpeg.input(processed_video_path).video
#     audio = ffmpeg.input(original_video_path).audio
#     ffmpeg.output(audio, video, processed_video_path, vcodec='copy', acodec='copy').run()


def merge_audio_with_ffmpeg_windows(video_path, audio_path, processed_video_path):
    """Use FFmpeg to merge the original audio with the processed video and overwrite the original"""
    print(f"Merging audio from '{audio_path}' to {video_path}")
    ffmpeg_command = f'ffmpeg -i "{video_path}" -i "{audio_path}" -c copy -map 0:v -map 1:a -shortest -y "{processed_video_path}"'
    os.system(ffmpeg_command)
    print(f"Audio merged")


def process_video(input_video_path, output_video_path):
    """Main function to process the video"""
    global brightness_average
    # Load the video
    video = cv2.VideoCapture(input_video_path)
    frames = []
    # Get the total number of frames and the frame rate
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_rate = video.get(cv2.CAP_PROP_FPS)
    tqdm.write(f"\nVideo in = {input_video_path}")
    tqdm.write(f"{total_frames = }, {frame_rate = }, {smoothing_factor = }")

    # Read all frames from the video
    success, frame = video.read()
    brightness_list = []
    with tqdm(total=total_frames, desc="Extracting frames from video", file=sys.stdout, colour=tqdm_color) as pbar:
        while success:
            frames.append(frame)
            if T_brightness__F_normalize:
                brightness_list.append(get_frame_brightness(frame))
            success, frame = video.read()
            pbar.update(1)
    video.release()
    brightness_average = np.mean(brightness_list)
    print(f'{brightness_average = }')

    # Normalize lighting for each frame
    if T_brightness__F_normalize:
        frames = [adjust_brightness(frame) for frame in tqdm(frames, desc="Adjusting the brightness", file=sys.stdout, colour=tqdm_color)]
    else:
        frames = [normalize_lighting(frame) for frame in tqdm(frames, desc="Normalizing the lighting", file=sys.stdout, colour=tqdm_color)]

    # Apply smoothing if required
    if bool(smoothing_factor):
        frames = smooth_lighting(frames)

    # Recombine the processed frames into a new video
    height, width, layers = frames[0].shape
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*video_codec), frame_rate, (width, height))
    for frame in tqdm(frames, desc="Writing processed frames to video", file=sys.stdout, colour=tqdm_color):
        out.write(frame)
    out.release()
    tqdm.write(f"Video processing complete. Output saved to {output_video_path}\n")


if __name__ == '__main__':
    videos = [r"C:\Users\eddy.a\Videos\ESP 01 (29-01-2023)\2023-01-29 _ 00-10-59.mp4",
              r"C:\Users\eddy.a\Videos\ESP 01 (29-01-2023)\2023-01-29 _ 00-08-52.mp4",
              r"C:\Users\eddy.a\Videos\ESP 01 (29-01-2023)\2023-01-29 _ 00-06-01.mp4",
              r"C:\Users\eddy.a\Videos\ESP 01 (29-01-2023)\2023-01-29 _ 00-02-35.mp4"][1:]
    # videos = [r"C:\Users\eddya\Videos\OG.mp4"]
    for input_video in videos:
        for video_codec in ['X264', 'mp4v']:
            output_video = input_video.replace(".mp4", f" ({video_codec}).mp4")
            merged_video = output_video.replace(").mp4", f" MERGED).mp4")
            process_video(input_video, output_video)
            merge_audio_with_ffmpeg_windows(input_video, output_video, merged_video)
            # merge_audio_with_ffmpeg_python(input_video, output_video)
