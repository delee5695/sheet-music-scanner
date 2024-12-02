from music21 import note, stream, dynamics, articulations, expressions
from PIL import Image
import random


def generate_note_image_with_variations(
    pitch, note_length, filename, add_variations=True
):
    """
    Generate an image of a single note with variations and crop it.
    :param pitch: Pitch of the note (e.g., 'C4', 'D#5')
    :param note_length: Length of the note ('whole', 'half', 'quarter', etc.)
    :param filename: Output filename for the image
    :param add_variations: Whether to add variations like dynamics, beams, or accidentals
    """
    s = stream.Stream()

    if note_length in ["eight", "sixteenth"]:
        # Add connected notes (beams)
        n1 = note.Note(pitch)
        n1.quarterLength = note_length
        n2 = note.Note(pitch)  # Another note for the beam
        n2.quarterLength = note_length
        n2.beams.fill("start")  # Start beam
        n1.beams.fill("stop")  # Stop beam
        s.append([n1, n2])
    else:
        # Add a single note
        n = note.Note(pitch)
        n.quarterLength = note_length
        if add_variations:
            # Add random accidental
            if random.choice([True, False]):
                n.pitch.accidental = random.choice(["sharp", "flat", "natural"])
            # Add random articulation
            if random.choice([True, False]):
                articulation = random.choice(
                    [articulations.Staccato(), articulations.Tenuto()]
                )
                n.articulations.append(articulation)
            # Add dynamics
            if random.choice([True, False]):
                dyn = random.choice(
                    [
                        dynamics.Dynamic("pp"),
                        dynamics.Dynamic("mf"),
                        dynamics.Dynamic("ff"),
                    ]
                )
                s.insert(0, dyn)  # Add dynamics at the beginning
        s.append(n)

    # Add other random elements
    if add_variations and random.choice([True, False]):
        s.append(expressions.Fermata())

    # Save to PNG
    temp_filename = f"{filename}_temp"
    s.write("lily.png", fp=temp_filename)

    # Crop the image to focus on the note
    crop_note(temp_filename + ".png", filename + ".png")


def crop_note(input_filename, output_filename):
    """
    Crop the generated note image to focus on the relevant area.
    :param input_filename: Full-sized image filename
    :param output_filename: Cropped image filename
    """
    with Image.open(input_filename) as img:
        left, top, right, bottom = 120, 15, 150, 70  # crop coordinates
        cropped_img = img.crop((left, top, right, bottom))
        cropped_img.save(output_filename)
        print(f"Cropped and saved {output_filename}")


if __name__ == "__main__":
    pitches = ["C4", "D4", "E4", "F4", "G4", "A4", "B4"]
    lengths = {
        "whole": 4.0,
        "half": 2.0,
        "quarter": 1.0,
        "eight": 0.5,
        "sixteenth": 0.25,
    }  # Note lengths

    for pitch in pitches:
        for name, length in lengths.items():
            filename = f"data/note_{pitch}_{name}"
            generate_note_image_with_variations(pitch, length, filename)
