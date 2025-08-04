import fitz  # PyMuPDF
from pdf2image import convert_from_path


rotate = [False, 270, r""]                          # [Enable, Rotation angle, Input file]
merge = [False, [r"", r""]]                         # [Enable, Input files]
split = [False, r"", [" - 01", " - 02", " - 03"]]   # [Enable, Input file, Output file suffixes]
extract_text = [False, r""]                         # [Enable, Input file]
convert_to_image = [False, r""]                     # [Enable, Input file]
output_file = r""


def rotate_pdf():
    doc = fitz.open(rotate[2])
    for page in doc:
        page.set_rotation(rotate[1])
    doc.save(output_file)
    doc.close()


def merge_pdfs():
    output = fitz.open()
    for file in merge[1]:
        src = fitz.open(file)
        output.insert_pdf(src)
        src.close()

    # Example of selecting specific pages (manual, as PyMuPDF doesn’t support ranges directly)
    # src1 = fitz.open(merge[1][0])
    # output.insert_pdf(src1, from_page=0, to_page=0)  # Page 1
    # src1.close()
    # src2 = fitz.open(merge[1][1])
    # output.insert_pdf(src2, from_page=2, to_page=2)  # Page 3
    # src2.close()

    output.save(output_file)
    output.close()


def split_pdf():
    input_pdf = fitz.open(split[1])
    total_pages = len(input_pdf)
    pages_per_split = len(split[2])

    for suffix, start in enumerate(range(0, total_pages, pages_per_split), split[2]):
        end = start + pages_per_split
        if end > total_pages:
            end = total_pages

        output_pdf = fitz.open()
        output_pdf.insert_pdf(input_pdf, from_page=start, to_page=end - 1)

        file = f"{output_file[:-4]}{suffix}{output_file[-4:]}"
        output_pdf.save(file)
        output_pdf.close()
    input_pdf.close()


def pdf_to_text():
    doc = fitz.open(extract_text[1])
    extracted_text = ""
    for page in doc:
        extracted_text += page.get_text()
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(extracted_text)
    doc.close()


def pdf_to_image():
    images = convert_from_path(convert_to_image[1])
    for i, image in enumerate(images):
        file = f"{output_file[:-4]}{i + 1:00}{output_file[-4:]}"
        image.save(file, "PNG")


if __name__ == '__main__':
    if merge[0]:
        merge_pdfs()
    if rotate[0]:
        rotate_pdf()
    if split[0]:
        split_pdf()
    if extract_text[0]:
        pdf_to_text()
    if convert_to_image[0]:
        pdf_to_image()
