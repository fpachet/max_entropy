import numpy as np
import mido
from scipy.optimize import minimize

def generate_sequence_mono_note(length=100):
    unique_notes = [60]
    sequence = [np.random.choice(unique_notes)]
        # Sample next note
    for _ in range(1, length):
        sequence.append(np.random.choice(unique_notes))
    return sequence
def generate_sequence_2_notes(length=100):
    unique_notes = [60, 62]
    sequence = []
        # Sample next note
    for _ in range(1, (int) (length/2)):
        sequence.append(unique_notes[0])
        sequence.append(unique_notes[1])
    return sequence

def generate_sequence_3_notes(length=100):
    unique_notes = [60, 62, 64]
    sequence = []
        # Sample next note
    for _ in range(1, (int) (length/3)):
        sequence.append(unique_notes[0])
        sequence.append(unique_notes[1])
        sequence.append(unique_notes[2])
    return sequence
def generate_sequence_arpeggios(length=100):
    unique_notes = [60, 62, 64, 65, 67, 69, 81]
    sequence = []
        # Sample next note
    start = 0
    for _ in range(1, (int) (length/4)):
        sequence.append(unique_notes[start % 7])
        sequence.append(unique_notes[(start + 1) % 7])
        sequence.append(unique_notes[(start + 2) % 7])
        sequence.append(unique_notes[(start + 4) % 7])
        start = start + 1
    return sequence

def save_to_midi(sequence, filename="generated.mid"):
    """Save the generated sequence as a MIDI file."""
    mid = mido.MidiFile()
    track = mido.MidiTrack()
    mid.tracks.append(track)
    for note in sequence:
        track.append(mido.Message('note_on', note=note, velocity=64, time=200))
        track.append(mido.Message('note_off', note=note, velocity=64, time=240))
    mid.save(filename)
    print(f"MIDI file saved as {filename}")


generated_sequence = generate_sequence_arpeggios(2000)
save_to_midi(generated_sequence, "../data/test_sequence_arpeggios.mid")
print("Generated MIDI Notes:", generated_sequence)
