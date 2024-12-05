"""Augment original data by cropping in different places"""

import os
from PIL import Image


def crop_note_with_variation(input_filename, output_filename, crop_id=0):
    """
    Crop the generated note image to focus on the relevant area.
    :param input_filename: Full-sized image filename
    :param output_filename: Cropped image filename
    """
    with Image.open(input_filename) as img:
        if crop_id == 0:
            left, top, right, bottom = 115, 10, 155, 50  # Adjust based on staff size
        elif crop_id == 1:
            left, top, right, bottom = 115, 5, 155, 45  # up
        elif crop_id == 2:
            left, top, right, bottom = 115, 15, 155, 55  # down
        elif crop_id == 3:
            left, top, right, bottom = 110, 10, 150, 50  # left
        else:
            left, top, right, bottom = 120, 10, 160, 50  # right
        cropped_img = img.crop((left, top, right, bottom))
        cropped_img.save(output_filename)
        print(f"Cropped and saved {output_filename}")


if __name__ == "__main__":
    png_files = []
    for filename in os.listdir("temp"):
        if filename.endswith(".png"):
            png_files.append(filename)
    DATA_POINTS = 0
    for file in png_files:
        for crop_id in range(5):
            output_file_name = file.split(".")[0] + f"_{crop_id}" + ".png"
            crop_note_with_variation(
                os.path.join("temp", file),
                os.path.join("data1", output_file_name),
                crop_id,
            )
            DATA_POINTS += 1
    print(DATA_POINTS)
