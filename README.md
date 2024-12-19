# music-note-classifier

We created a CNN model that can detect the pitch and length of a musical note from an image of a single note on the staff. To do this, we created our own dataset of images of notes using LilyPond, augmented them, and trained a CNN model to categorize them. This repo contains our code to generate data, augment the data, and train our model, as well as our final model weights.

### Website
Visit our website here: https://github.com/dongim04/musical-note-classifier

The website contains:
- A demo where you can run our model on your own image of a note (must download and host the website on your local machine to do this)
- More information about our model (statistics, how we developed it, etc)

### What's in this repo?
- An annotated Colab notebook explaining all of our data generation, augmentation, and model training code ([note_classifier.ipynb](https://colab.research.google.com/drive/1Hfccm4PxE03oROSF8VCTFl2dbOS89iVS?usp=sharing)).
- The dataset we generated (dataset.zip).
- Separate code files to generate note data, augment it, and train the model on it (generate_data.py, augment_data.py, train_model.py).
- The trained model weights and encoder for them (model_weights.onnx, encoder.pkl).
- Code to load the model weights and run the model on some of your own images (use_model.py).
- Colab notebooks containing the code for separate pitch and length models that we also tried but didn't end up developing
  ([pitch_model.ipynb](https://colab.research.google.com/drive/1TPdC_ecRRHT3PCibH7gpcuPoYSRZiJfY))
  ([length_model.ipynb](https://colab.research.google.com/drive/1a8o2D7-Ia2m7ToX8gd5USg6IC7ICE07T))

### Running our code
Data generation:
- If you would like to run our data generation code, you will need to download Lilypond (https://lilypond.org/doc/v2.23/Documentation/web/download)

For the rest of our code, you can either run it in the Colab notebook or on your local machine.

If you want to run it on your local machine:
1. Clone this repository:
   ```bash
   git clone https://github.com/delee5695/sheet-music-scanner.git
   cd sheet-music-scanner
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
