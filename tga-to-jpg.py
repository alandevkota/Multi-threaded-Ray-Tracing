
from PIL import Image

def convert_tga_to_jpg(tga_file, jpg_file):
  """Converts a TGA file to a JPG file.

  Args:
    tga_file: The path to the TGA file.
    jpg_file: The path to the JPG file.
  """

  with Image.open(tga_file) as image:
    image.save(jpg_file, "JPEG")

if __name__ == "__main__":
  convert_tga_to_jpg("output.tga", "output.jpg")