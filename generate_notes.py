from music21 import note, stream, dynamics, articulations, expressions
from PIL import Image
import os


def generate_note_image_with_variations(pitch, name, note_length, filename, variations):
    """
    Generate an image of a single note with variations and crop it.
    :param pitch: Pitch of the note (e.g., 'C4', 'D#5')
    :param note_length: Length of the note ('whole', 'half', 'quarter', etc.)
    :param filename: Output filename for the image
    :param variations: Dict containing variations (dynamics, articulation, accidental, fermata)
    """
    s = stream.Stream()

    if name in ["eighth", "16th"]:
        # Add connected notes (beams)
        n1 = note.Note(pitch, quarterLength=note_length)
        n2 = note.Note(pitch, quarterLength=note_length)  # Another note for the beam
        n1.beams.fill(name, type="start")  # Start beam
        n2.beams.fill(name, type="stop")  # Stop beam
        s.append([n1, n2])
    else:
        # Add a single note
        n = note.Note(pitch)
        n.quarterLength = note_length

        # Apply variations
        if variations.get("accidental"):
            n.pitch.accidental = variations["accidental"]
        if variations.get("articulation"):
            n.articulations.append(variations["articulation"])
        s.append(n)

    # Save to PNG
    temp_filename = os.path.join("temp", filename)
    s.write("lily.png", fp=temp_filename)

    # Crop the image to focus on the note
    crop_note(f"{temp_filename}.png", os.path.join("data", f"{filename}.png"))


def crop_note(input_filename, output_filename):
    """
    Crop the generated note image to focus on the relevant area.
    :param input_filename: Full-sized image filename
    :param output_filename: Cropped image filename
    """
    with Image.open(input_filename) as img:
        left, top, right, bottom = 120, 15, 150, 70  # Adjust based on staff size
        cropped_img = img.crop((left, top, right, bottom))
        cropped_img.save(output_filename)
        print(f"Cropped and saved {output_filename}")


if __name__ == "__main__":
    pitches = ["C4", "D4", "E4", "F4", "G4", "A4", "B4"]  # Example pitches
    lengths = {
        "whole": 4.0,
        "half": 2.0,
        "quarter": 1.0,
        "eighth": 0.5,
        "16th": 0.25,
    }  # Note lengths
    #     variation_id = 0
    #     variations = {
    #         "articulation": articulations.Staccato(),
    #         "accidental": None,
    #     }
    #     for pitch in pitches:
    #         for name, length in lengths.items():
    #             variation_id += 1
    #             filename = f"note_{pitch}_{name}_variation_{variation_id}"
    #             generate_note_image_with_variations(
    #                 pitch, name, length, filename, variations
    #             )

    # Define all variations
    # articulation_variations = [None, articulations.Staccato(), articulations.Tenuto()]
    articulation_variations = [None]
    accidental_variations = [None, "sharp", "flat"]

    # Generate combinations of variations
    for pitch in pitches:
        for name, length in lengths.items():
            variation_id = 0
            if name in ["eighth", "16th"]:
                variations = {
                    "articulation": None,
                    "accidental": None,
                }
                filename = f"note_{pitch}_{name}_variation_{variation_id}"
                generate_note_image_with_variations(
                    pitch, name, length, filename, variations
                )
            else:
                for art in articulation_variations:
                    for acc in accidental_variations:
                        variations = {
                            "articulation": art,
                            "accidental": acc,
                        }
                        filename = f"note_{pitch}_{name}_variation_{variation_id}"
                        generate_note_image_with_variations(
                            pitch, name, length, filename, variations
                        )
                        variation_id += 1
