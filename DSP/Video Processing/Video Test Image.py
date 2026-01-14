import cv2
import numpy as np


fourcc = cv2.VideoWriter_fourcc(*"mp4v")


def ramp_up_brightness(writer):
    black_pct, ramp_pct, white_pct = frame_distribution
    black_frames = int(total_frames * black_pct / 100)
    ramp_frames  = int(total_frames * ramp_pct  / 100)
    white_frames = total_frames - black_frames - ramp_frames
    print(f'{black_frames = }, {ramp_frames = }, {white_frames = }')

    for frame_idx in range(total_frames):
        if frame_idx < black_frames:
            intensity = 0
        elif frame_idx < black_frames + ramp_frames:
            ramp_idx = frame_idx - black_frames
            intensity = int(255 * ramp_idx / (ramp_frames - 1))
        else:
            intensity = 255

        red = intensity if pick_color in [0, 3] else 0
        green = intensity if pick_color in [1, 3] else 0
        blue = intensity if pick_color in [2, 3] else 0
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:, :, 2] = red
        frame[:, :, 1] = green
        frame[:, :, 0] = blue

        text = f"RGB: {red}, {green}, {blue}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 2
        thickness = 4
        text_color = (255, 255, 255) if sum([red, green, blue]) / 3 < 128 else (0, 0, 0)

        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_x = (width - text_size[0]) // 2
        text_y = (height + text_size[1]) // 2

        cv2.putText(frame, text, (text_x, text_y), font, font_scale, text_color, thickness, cv2.LINE_AA)
        writer.write(frame)


if __name__ == '__main__':
    width, height = 1920, 1080
    fps = 30
    duration = 10
    total_frames = fps * duration
    color = ["White", "Red", "Green", "Blue"]
    pick_color = 0
    frame_distribution = [10, 80, 10]
    assert sum(frame_distribution) == 100

    output_file = rf"Test 01 - {color[pick_color]}.mp4"

    writer = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
    ramp_up_brightness(writer)
    writer.release()
    print("Video written to", output_file)
