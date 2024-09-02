import sys
import cv2
from tqdm import tqdm


video_codec = ['H264', 'X264', 'mp4v'][0]
tqdm_color = 'cyan'


def normalize_lighting(frame):
    """Function to normalize lighting using histogram equalization"""
    # Convert the frame to YUV (Y: brightness, U, V: color channels)
    yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
    # Equalize the histogram of the Y channel (brightness)
    yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
    # Convert back to BGR
    result_frame = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
    return result_frame


def smooth_lighting(frames, alpha):
    """Function to smooth lighting transitions between frames"""
    smoothed_frames = [frames[0]]  # Start with the first frame
    for i in tqdm(range(1, len(frames)), desc="Smoothing lighting transitions", file=sys.stdout, colour=tqdm_color):
        # Blend the previous frame with the current one
        blended_frame = cv2.addWeighted(smoothed_frames[i - 1], alpha, frames[i], 1 - alpha, 0)
        smoothed_frames.append(blended_frame)
    return smoothed_frames


def process_video(input_video_path, output_video_path, apply_smoothing, alpha):
    """Main function to process the video"""
    # Load the video
    video = cv2.VideoCapture(input_video_path)
    frames = []
    # Get the total number of frames and the frame rate
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_rate = video.get(cv2.CAP_PROP_FPS)
    tqdm.write(f"\nVideo in = {input_video_path}")
    tqdm.write(f"{total_frames = }, {frame_rate = }, {apply_smoothing = }, {alpha = }")

    # Read all frames from the video
    success, frame = video.read()
    with tqdm(total=total_frames, desc="Extracting frames from video", file=sys.stdout, colour=tqdm_color) as pbar:
        while success:
            frames.append(frame)
            success, frame = video.read()
            pbar.update(1)
    video.release()

    # Normalize lighting for each frame
    normalized_frames = [normalize_lighting(frame) for frame in tqdm(frames, desc="Normalizing the lighting", file=sys.stdout, colour=tqdm_color)]

    # Apply smoothing if required
    if apply_smoothing:
        smoothed_frames = smooth_lighting(normalized_frames, alpha)
    else:
        smoothed_frames = normalized_frames

    # Recombine the processed frames into a new video
    height, width, layers = frames[0].shape
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*video_codec), frame_rate, (width, height))
    for frame in tqdm(smoothed_frames, desc="Writing processed frames to video", file=sys.stdout, colour=tqdm_color):
        out.write(frame)
    out.release()
    tqdm.write(f"Video processing complete. Output saved to {output_video_path}\n")


if __name__ == '__main__':
    videos = [r"C:\Users\eddya\Videos\Table LIGHT.mp4", r"C:\Users\eddya\Videos\Table DARK.mp4",
              r"C:\Users\eddya\Videos\Bed LIGHT.mp4", r"C:\Users\eddya\Videos\Bed DARK.mp4"]
    for input_video in videos:
        for alpha in [0, 0.1, 0.5, 0.9, 1]:
            output_video = input_video.replace(".mp4", f" alpha={int(alpha * 100)}%.mp4")
            process_video(input_video, output_video, apply_smoothing=bool(alpha), alpha=alpha)
