from rembg import remove
from PIL import Image

input_path = r""
output_path = r""

file_in = Image.open(input_path)
file_out = remove(file_in)
file_out.save(output_path)
