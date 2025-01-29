"""
This script is solely designed to compare two pictures.
If you have two PDFs, this will convert them to pictures first.

So, if you want to compare LTspice schematics, first export them to pictures or PDFs.
For the best results:
  1. File → disable "Print Monochrome".
  2. File → Print Setup → Name → choose "Microsoft Print to PDF".
  3. File → Print Setup → Paper → choose "A3" (should be big enough for a nice resolution).
  4. File → Print. This will export the schematic to a PDF.
"""


from PIL import Image, ImageChops, ImageDraw
import fitz


color_missing_in_after = "green"
color_missing_in_before = "red"


def pdf_to_image(pdf_path, page=0):
    doc = fitz.open(pdf_path)
    page = doc.load_page(page)

    # PyMuPDF's default is around 72 DPI, so scale factor is dpi / 72
    zoom_x = dpi / 72.0
    zoom_y = dpi / 72.0
    matrix = fitz.Matrix(zoom_x, zoom_y)

    pix = page.get_pixmap(matrix=matrix)
    image_path = pdf_path.replace('.pdf', '.png')
    pix.save(image_path)
    return image_path


def compare_images(img1_path, img2_path, diff_path):
    img1 = Image.open(img1_path).convert("RGB")
    img2 = Image.open(img2_path).convert("RGB")

    # Ensure both images have the same size
    img1 = img1.resize(img2.size)
    diff = ImageChops.difference(img1, img2)

    # Convert difference image to grayscale
    diff_gray = diff.convert("L")

    # Create a drawing context on the original image
    draw = ImageDraw.Draw(img1)

    # Set a threshold for detecting changes
    threshold = 50
    width, height = diff.size

    for x in range(width):
        for y in range(height):
            diff_pixel = diff_gray.getpixel((x, y))
            if diff_pixel > threshold:
                pixel1 = img1.getpixel((x, y))
                pixel2 = img2.getpixel((x, y))
                if sum(pixel1) < sum(pixel2):
                    color = color_missing_in_before
                    draw.ellipse((x - 5, y - 5, x + 5, y + 5), outline=color, width=2)

    for x in range(width):
        for y in range(height):
            diff_pixel = diff_gray.getpixel((x, y))
            if diff_pixel > threshold:
                pixel1 = img1.getpixel((x, y))
                pixel2 = img2.getpixel((x, y))
                if sum(pixel1) > sum(pixel2):
                    color = color_missing_in_after
                    draw.ellipse((x - 5, y - 5, x + 5, y + 5), outline=color, width=2)

    img1.save(diff_path)  # Save the highlighted image


if __name__ == '__main__':
    path_folder = r""
    if path_folder is None or path_folder == "":
        path_folder = os.path.dirname(os.path.realpath(__file__))
    dpi = 300
    convert_to_png = False
    path_file_1 = f"{path_folder}\\Before.{"pdf" if convert_to_png else "png"}"
    path_file_2 = f"{path_folder}\\After.{"pdf" if convert_to_png else "png"}"
    path_diff_file = f"{path_folder}\\Diff.png"

    if convert_to_png:
        path_file_1 = pdf_to_image(path_file_1)
        path_file_2 = pdf_to_image(path_file_2)
    compare_images(path_file_1, path_file_2, path_diff_file)
    print(f'Difference image saved as: {path_diff_file}')
