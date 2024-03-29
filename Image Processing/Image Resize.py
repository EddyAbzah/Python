"""
   interpolation_type   |                                        Interpretation                                          |   Shrinkage   |   Enlargement
-------------------------------------------------------------------------------------------------------------------------|---------------|-----------------
      INTER_LINEAR      |                 The standard bi-linear interpolation; ideal for enlarged images                |    It's OK    |      blurry
      INTER_NEAREST     |           The nearest neighbor interpolation; fast to run, but creates blocky images           |    Horrible   |     too sharp
       INTER_AREA       |                 The interpolation for the pixel area, which scales down images                 |    perfect    |     too sharp
       INTER_CUBIC      |      The bicubic interpolation with 4×4-pixel neighborhood; slow to run, but high-quality      |    Horrible   |      perfect
     INTER_LANCZOS4     |   The Lanczos interpolation with 8×8-pixel neighborhood; slowest to run, but highest quality   |    Horrible   |      perfect
"""

import cv2

input_path = r""
scale_factor = 1
output_path = f" ({scale_factor = }).".join(input_path.rsplit('.', 1))
interpolation_types = [cv2.INTER_LINEAR, cv2.INTER_NEAREST, cv2.INTER_AREA, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4]
interpolation_pick = ["all", "best", 0][1]

image = cv2.imread(input_path)
if interpolation_pick == "best":
    if scale_factor < 1:
        interpolation = cv2.INTER_AREA
    elif scale_factor > 1:
        interpolation = cv2.INTER_LANCZOS4
    else:
        interpolation = cv2.INTER_LINEAR
    zoomed_image = cv2.resize(src=image, dsize=None, fx=scale_factor, fy=scale_factor, interpolation=interpolation)
    cv2.imwrite(output_path, zoomed_image)
elif interpolation_pick == "all":
    for index_out, interpolation in enumerate(interpolation_types):
        zoomed_image = cv2.resize(src=image, dsize=None, fx=scale_factor, fy=scale_factor, interpolation=interpolation)
        cv2.imwrite(f"{output_path.rsplit('.', 1)[0]} {index_out:02}.{output_path.rsplit('.', 1)[-1]}", zoomed_image)
else:
    zoomed_image = cv2.resize(src=image, dsize=None, fx=scale_factor, fy=scale_factor, interpolation=interpolation_types[interpolation_pick])
    cv2.imwrite(output_path, zoomed_image)
