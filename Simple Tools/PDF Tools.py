"""
Needs PyPDF2 version 2.12.1 or lower to work.

Check:
print(PyPDF2.__version__)

Install:
pip install PyPDF2==2.12.1
"""


import PyPDF2
from pdf2image import convert_from_path


rotate = [False, 270, r""]                          # [Enable, Rotation, Input file]
merge = [False, [r"", r""]]                         # [Enable, Input files]
split = [False, r"", [" - 01", " - 02", " - 03"]]   # [Enable, Input file, Output file suffixes]
extract_text = [False, r""]                         # [Enable, Input file]
convert_to_image = [False, r""]                     # [Enable, Input file]
output_file = r""


def rotate_pdf():
    reader = PyPDF2.PdfReader(rotate[2])
    writer = PyPDF2.PdfWriter()
    for page in reader.pages:
        page.rotate(rotate[1])
        writer.add_page(page)
    with open(output_file, "wb") as pdf_out:
        writer.write(pdf_out)


def merge_pdfs():
    merger = PyPDF2.PdfMerger()

    for file in merge[1]:
        merger.append(file)
    # Or, to select specific pages (zero-based index + non-inclusive end index):
    # merger.append(merge[1][0], pages=(0, 1))
    # merger.append(merge[1][1], pages=(2, 3))

    merger.write(output_file)
    merger.close()


def split_pdf():
    input_pdf = PyPDF2.PdfFileReader(split[1])
    total_pages = input_pdf.getNumPages()
    pages_per_split = len(split[2])
    
    for suffix, start in enumerate(range(0, total_pages, pages_per_split), split[2]):
        end = start + pages_per_split
        if end > total_pages:
            end = total_pages

        output_pdf = PyPDF2.PdfFileWriter()
        for page in range(start, end):
            output_pdf.addPage(input_pdf.getPage(page))

        file = f"{output_file[:-4]}{suffix}{output_file[-4:]}"
        with open(file, "wb") as f:
            output_pdf.write(f)


def pdf_to_text():
    input_pdf = PyPDF2.PdfFileReader(extract_text[1])
    extracted_text = ""
    for page in range(input_pdf.getNumPages()):
        extracted_text += input_pdf.getPage(page).extractText()
    with open(output_file, "wb") as f:
        f.write(extracted_text.encode('utf8'))


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
