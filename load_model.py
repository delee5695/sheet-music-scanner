import os
import numpy as np
import tensorflow as tf
from PIL import Image
import pickle

model = tf.keras.models.load_model('/path/to/model/one_step_sheet_music_model.onnx')

with open('/path/to/encoder/encoder.pkl', 'rb') as file:
    encoder = pickle.load(file)

examples_folder = 'folder containing the images you want to test the model on'

for file_name in os.listdir(examples_folder):
    file_path = os.path.join(examples_folder, file_name)

    image = Image.open(file_path)
    image_array = np.array(image.convert('L').resize((64, 64))) / 255.0
    image_array = np.expand_dims(image_array, axis=0)  # Shape becomes (1, 64, 64)
    image_array = np.expand_dims(image_array, axis=-1)  # Shape becomes (1, 64, 64, 1)

    predictions = model.predict(image_array)
    predicted_class_index = np.argmax(predictions)
    predicted_label = encoder.inverse_transform([predicted_class_index])
    confidence = predictions[0][predicted_class_index]

    print(f"Actual label: {file_name[:-4]}")
    print(f"Predicted label: {predicted_label[0]}")
    print(f"Confidence: {confidence:.2f}")
    print('-' * 40)
