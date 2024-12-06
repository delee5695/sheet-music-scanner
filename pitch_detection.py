import torch
import torch.nn as nn
from torchvision import datasets, transforms
from PIL import Image
import os


# CNN Model
class PitchClassifier(nn.Module):
    def __init__(self, num_classes):
        super(PitchClassifier, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),  # 56x56 -> 56x56
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 56x56 -> 28x28
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 14x14
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 7x7
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                64 * 7 * 7, 128
            ),  # channels of last output layer * output dimensions, basically number of output parameters from convolutions
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x


IMAGE_SIZE = 56  # Desired size for the square images
DATA_DIR = "./data_pitch"

# Define the same transformations used during training
transform = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), antialias=True),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ]
)
train_dataset = datasets.ImageFolder(
    root=os.path.join(DATA_DIR, "train"), transform=transform
)


# Function to load the trained model
def load_model(model_path, num_classes):
    model = PitchClassifier(num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model


# Function to predict the class of an input image
def predict_image(model, image_path, class_labels):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load and preprocess the image
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0).to(device)  # Add batch dimension

    # Make prediction
    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()

    return class_labels[predicted_class], confidence


# Usage Example
if __name__ == "__main__":
    # Path to the saved model
    model_path = "./pitch_classifier.pth"  # Replace with your model's path

    # Load model and class labels
    class_labels = train_dataset.classes  # Use the same labels as during training
    model = load_model(model_path, len(class_labels))

    # Path to an input image
    image_path = (
        "./augmented_data/note_D4_half_variation_1.png"  # Replace with your test image
    )

    # Get prediction
    predicted_label, confidence = predict_image(model, image_path, class_labels)
    print(f"Predicted label: {predicted_label}, Confidence: {confidence:.4f}")
