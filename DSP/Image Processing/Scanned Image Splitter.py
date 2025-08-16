import os
import re
import cv2
import numpy as np
from PIL import Image, ImageEnhance


def remove_white_borders(img, threshold=235):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
    coords = cv2.findNonZero(thresh)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        return img[y:y+h, x:x+w]
    else:
        return img  # fallback: return as-is


def enhance_image(pil_img, brightness, contrast):
    enhancer = ImageEnhance.Contrast(pil_img)
    pil_img = enhancer.enhance(contrast)
    enhancer = ImageEnhance.Brightness(pil_img)
    pil_img = enhancer.enhance(brightness)
    return pil_img


def white_balance(img):
    result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])
    result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
    return cv2.cvtColor(result, cv2.COLOR_LAB2BGR)


def split_image(img, suffix):
    h, w = img.shape[:2]
    parts = []

    if suffix == '2H':
        parts = [img[0:h//2, :], img[h//2:, :]]
    elif suffix == '3H':
        parts = [img[int(y0 * h):int(y1 * h), :] for y0, y1 in split_3h]
    elif suffix in ['2H2V', '4H']:
        half_h, half_w = h // 2, w // 2
        parts = [img[0:half_h, 0:half_w], img[0:half_h, half_w:],
                 img[half_h:, 0:half_w], img[half_h:, half_w:]]
        if suffix == '4H':
            # Rotate each to landscape
            parts = [cv2.rotate(p, cv2.ROTATE_90_CLOCKWISE) for p in parts]
    else:
        parts = [img]  # No split

    return parts


def load_image(path):
    pil_img = Image.open(path).convert('RGB')
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def process_image(cropped):
    wb = white_balance(cropped)
    pil_img = Image.fromarray(cv2.cvtColor(wb, cv2.COLOR_BGR2RGB))
    return pil_img


if __name__ == '__main__':
    folder_in = r""
    folder_out = r""
    # brightness_list = [1]
    brightness_list = [1, 1.1]
    # contrast_list = [1]
    contrast_list = [1, 1.1]

    split_3h = [(0, 0.35), (0.3, 0.7), (0.65, 1)]

    filter_files = re.compile(r'.*\.jpg$', re.IGNORECASE)
    all_files = [folder_in + "\\" + file for file in os.listdir(folder_in) if filter_files.match(file)]
    all_files.sort()
    os.makedirs(folder_out, exist_ok=True)

    for index, file in enumerate(all_files, start=1):
        file_basename = os.path.basename(file)
        if "_" in file_basename:
            file_suffix = os.path.splitext(file_basename)[0].split("_")[1]
            file_basename = file_basename.split("_")[0]
        else:
            file_suffix = ""
        print(f"Image {index:02}: {file_basename}")

        for brightness, contrast in zip(brightness_list, contrast_list):
            image = load_image(file)
            segments = split_image(image, file_suffix)
            for i, part in enumerate(segments):
                cropped_part = remove_white_borders(part, threshold=150)
                processed_part = process_image(cropped_part)
                enhanced_part = enhance_image(processed_part, brightness, contrast)
                if len(segments) > 1:
                    out_path = os.path.join(folder_out, f"{os.path.splitext(file_basename)[0]} _ {i + 1:03}")
                else:
                    out_path = os.path.join(folder_out, f"{os.path.splitext(file_basename)[0]}")
                out_path += f" (b-c={brightness:.1f}-{contrast:.1f}).jpg"
                enhanced_part.save(out_path)
                print(f"Image {index:02} Saved: {out_path}")
