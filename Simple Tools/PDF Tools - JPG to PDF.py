from PIL import Image
from reportlab.lib.pagesizes import A4, A3, landscape, portrait
from reportlab.pdfgen import canvas
import os


def create_pdf(folder_path, output_pdf, paper_size=A4, images_per_page=1, add_page_number=True, margin=10, orientation="Portrait"):
    paper_size = portrait(paper_size) if orientation == "Portrait" else landscape(paper_size)
    image_files = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith((".jpg", ".jpeg"))])
    c = canvas.Canvas(output_pdf, pagesize=paper_size)
    width, height = paper_size
    margin = margin * 2.83465  # Convert mm to points

    positions = {
        1: [(margin, margin, width - margin, height - margin)],
        2: [(margin, height // 2, width - margin, height - margin), (margin, margin, width - margin, height // 2)],
        4: [(margin, height // 2, width // 2 - margin, height - margin), (width // 2, height // 2, width - margin, height - margin),
            (margin, margin, width // 2 - margin, height // 2 - margin), (width // 2, margin, width - margin, height // 2 - margin)]
    }

    positions = positions.get(images_per_page, positions[1])
    count = 0
    page_number = 1
    total_pages = (len(image_files) + images_per_page - 1) // images_per_page

    for img_file in image_files:
        img = Image.open(img_file)
        img_width, img_height = img.size
        img_ratio = img_width / img_height

        if count % images_per_page == 0 and count > 0:
            if add_page_number:
                c.setFont("Helvetica", 12)
                c.drawString(width - 50, 30, f"{page_number} / {total_pages}")
            c.showPage()
            page_number += 1

        pos = positions[count % images_per_page]
        x1, y1, x2, y2 = pos
        target_width, target_height = x2 - x1, y2 - y1
        target_ratio = target_width / target_height

        if img_ratio > target_ratio:
            new_width = target_width
            new_height = int(target_width / img_ratio)
        else:
            new_height = target_height
            new_width = int(target_height * img_ratio)

        x_offset = x1 + (target_width - new_width) // 2
        y_offset = y1 + (target_height - new_height) // 2

        c.drawImage(img_file, x_offset, y_offset, new_width, new_height)
        count += 1

    if add_page_number:
        c.setFont("Helvetica", 12)
        c.drawString(width - 50, 30, f"{page_number} / {total_pages}")

    c.save()


if __name__ == "__main__":
    paper_size = {"A3": A3, "A4": A4}["A3"]
    orientation = ["Portrait", "Landscape"][1]
    images_per_page = 4
    margin = 1  # in mm
    add_page_number = True

    folder_path = r""
    output_pdf = rf"Output {paper_size} {orientation} {images_per_page} {margin}mm.pdf"

    create_pdf(folder_path, output_pdf, paper_size, images_per_page, add_page_number, margin, orientation)
