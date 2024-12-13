"""Augment original data by cropping in different places"""

import os
from PIL import Image
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelEncoder
import numpy as np
import os
from tqdm import tqdm


def crop_note_with_variation(input_filename, output_filename, crop_id=0):
    """
    Crop the generated note image to focus on the relevant area.
    :param input_filename: Full-sized image filename
    :param output_filename: Cropped image filename
    """
    lower_notes = ["A3", "B3", "C4", "D4", "E4", "F4", "G4", "A4", "B4"]

    # Check if any of the notes are in the filename

    with Image.open(input_filename) as img:
        # if any(note in filename for note in lower_notes):
        l, t, r, b = 110, 10, 170, 70
        if crop_id == 0:
            left, top, right, bottom = l, t, r, b  # original
        elif crop_id == 1:
            left, top, right, bottom = l, t - 5, r, b - 5  # up
        elif crop_id == 2:
            left, top, right, bottom = l, t + 5, r, b + 5  # down
        elif crop_id == 3:
            left, top, right, bottom = l - 5, t, r - 5, b  # left
        else:
            left, top, right, bottom = l, t + 5, r, b + 5  # right

def augment_data(input_foldername, output_foldername):
    file_names = os.listdir(input_foldername)

    for file_name in file_names:
        if file_name.endswith("(1).png"):
            file_names.remove(file_name)

    pitch_labels = []
    length_labels = []

    for file_name in file_names:
        pitch_labels.append(file_name[5:7])
        length_labels.append(file_name[8:-18])

    file_names = np.char.add(input_foldername, file_names)
    file_names = np.array(file_names)
    pitch_labels = np.array(pitch_labels)
    length_labels = np.array(length_labels)

    images = []

    for path in tqdm(file_names, desc="Processing images"):
        try:
            image = Image.open(path)
            image = image.convert('L')  # 'L' for grayscale
            image = image.resize((64, 64))
            image_array = np.array(image) / 255.0
            images.append(image_array)

        except Exception as e:
            print(f"Error loading image {path}: {e}")

    images = np.array(images)

    num_pitch_classes = len(set(pitch_labels))  # (C, D, E, F, G, A, B)
    num_length_classes = len(set(length_labels))  # (whole, half, quarter, eighth, 16th)
    num_classes = num_pitch_classes * num_length_classes

    images = np.expand_dims(images, axis=-1)  # Shape: (n, 64, 64, 1)

    combined_labels = [p + l for p, l in zip(pitch_labels, length_labels)]
    encoder = LabelEncoder()
    combined_labels_encoded = encoder.fit_transform(combined_labels)

    labels = to_categorical(combined_labels_encoded, num_classes=num_classes) # Shape: (n, num_classes)

    augmented_augmented_data_dir = output_foldername
    os.makedirs(augmented_augmented_data_dir, exist_ok=True)

    datagen = ImageDataGenerator(
        rotation_range=3,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode="nearest"
    )

    augmentations_per_image = 30
    augmented_images = []
    augmented_labels = []

    # Generate augmented images
    for i, (image, label) in enumerate(zip(images, labels)):
        for batch in datagen.flow(
            np.expand_dims(image, axis=0),  # Shape: (1, 64, 64, 1)
            batch_size=1,
            save_to_dir=augmented_augmented_data_dir,
            save_prefix=f"{file_names[i][67:-4]}",
            save_format="png"
        ):
            augmented_images.append(batch[0])
            augmented_labels.append(label)

            if len(augmented_images) % augmentations_per_image == 0:
                break

    all_images = np.concatenate((images, np.array(augmented_images)), axis=0)
    all_labels = np.concatenate((labels, np.array(augmented_labels)), axis=0)

if __name__ == "__main__":
    png_files = []
    for filename in os.listdir("temp"):
        if filename.endswith(".png"):
            png_files.append(filename)
    os.makedirs("data", exist_ok=True
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
