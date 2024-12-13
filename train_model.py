import tensorflow as tf
from tensorflow.keras import layers, models, Input, Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
import os
from PIL import Image
from tqdm import tqdm  # for progress bar
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt

# all_images = ?????
# all_labels = ?????

train_images, temp_images, train_labels, temp_labels = train_test_split(
    all_images, all_labels, test_size=0.3, random_state=42
)

val_images, test_images, val_labels, test_labels = train_test_split(
    temp_images, temp_labels, test_size=0.5, random_state=42
)

def create_advanced_cnn_model_with_l2(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape)

    # Block 1
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.001))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)

    # Block 2
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)

    # Block 3
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)

    # Attention Mechanism (SE Block)
    se = layers.GlobalAveragePooling2D()(x)
    se = layers.Dense(128 // 16, activation='relu', kernel_regularizer=l2(0.001))(se)  # Bottleneck
    se = layers.Dense(128, activation='sigmoid', kernel_regularizer=l2(0.001))(se)
    x = layers.multiply([x, se])  # Scale the feature maps

    # Block 4
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)

    # Global Average Pooling and Fully Connected Layers
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu', kernel_regularizer=l2(0.001))(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation='relu', kernel_regularizer=l2(0.001))(x)
    x = layers.Dropout(0.5)(x)

    # Output
    outputs = layers.Dense(num_classes, activation='softmax', kernel_regularizer=l2(0.001), name='output')(x)

    model = models.Model(inputs, outputs)
    return model

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

lr_scheduler = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    verbose=1
)

input_shape = (64, 64, 1)
model = create_advanced_cnn_model_with_l2(input_shape, num_classes) # create_advanced_cnn_model_with_l2
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

train_generator = datagen.flow(train_images, train_labels, batch_size=32)
val_generator = (val_images, val_labels)

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=500,
    callbacks=[early_stopping, lr_scheduler] # lr_scheduler
)



## Metrics

train_loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(train_loss) + 1)

plt.figure(figsize=(8, 6))
plt.plot(epochs, train_loss, label='Training Loss')
plt.plot(epochs, val_loss, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)
plt.show()

test_loss, test_accuracy = model.evaluate(test_images, test_labels, verbose=1)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

# Evaluate the model and get predictions
# test_loss, test_accuracy = model.evaluate(test_images, test_labels, verbose=1)
predicted_probs = model.predict(test_images)
predicted_classes = np.argmax(predicted_probs, axis=1)
true_classes = np.argmax(test_labels, axis=1)

# Print the test loss and accuracy
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

# Compute the confusion matrix
conf_matrix = confusion_matrix(true_classes, predicted_classes)
print("\nConfusion Matrix:")
print(conf_matrix)

# Compute additional metrics
print("\nClassification Report:")
print(classification_report(true_classes, predicted_classes, digits=4))

# Optionally, calculate and print additional individual metrics
accuracy = accuracy_score(true_classes, predicted_classes)
print(f"Accuracy: {accuracy}")
