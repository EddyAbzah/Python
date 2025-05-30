import os
from PIL import Image, ImageDraw, ImageFont


template_path = r"C:\Users\eddya\OneDrive\מסמכים\Python\DSP\Other\Template.jpg"
output_dir = r"C:\Users\eddya\Music\Various Languages - Deutsche Sprache"
labels = [""]

font = ImageFont.truetype(r"C:\Windows\Fonts\lhandw.ttf", 25)
rectangle = {"fill": (0, 0, 0, 100), "outline": "white"}

template = Image.open(template_path)
for label in labels:
    img = template.copy()
    draw = ImageDraw.Draw(img)

    text_bbox = draw.textbbox((0, 0), label, font=font)
    text_w = text_bbox[2] - text_bbox[0]
    text_h = text_bbox[3] - text_bbox[1]

    x = (img.width - text_w) // 2
    y = (img.height - text_h) // 2

    rectangle_padding = 5
    rectangle_coordinates = [x - rectangle_padding, y - rectangle_padding * 2, x + text_w + rectangle_padding, y + text_h + rectangle_padding * 4]

    # Create overlay for transparency
    overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)

    # Draw filled rectangle with transparency
    overlay_draw.rectangle(rectangle_coordinates, fill=rectangle["fill"], outline=rectangle["outline"], width=2)
    img = Image.alpha_composite(img.convert("RGBA"), overlay)
    draw = ImageDraw.Draw(img)
    draw.text((x, y), label, fill="white", font=font)

    filename = os.path.join(output_dir, f"{label}.png")
    img.save(filename)

print("✅ Images generated and saved in:", output_dir)
