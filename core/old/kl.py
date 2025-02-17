import mido
import numpy as np
import random
from scipy.stats import entropy
from collections import defaultdict


# Extract note and transition distributions for all distances 1 ≤ k ≤ max_k
def extract_midi_statistics(midi_file, max_k=10):
    """
    Extracts:
    - note_distribution: {note -> probability}
    - transition_distributions: {k -> {(note1, note2) -> probability}}

    The transition distributions are stored separately for all 1 ≤ k ≤ max_k.
    """
    mid = mido.MidiFile(midi_file)

    note_counts = defaultdict(int)
    transition_counts = {k: defaultdict(int) for k in range(1, max_k + 1)}
    total_notes = 0
    note_sequence = []

    for track in mid.tracks:
        for msg in track:
            if msg.type == 'note_on' and msg.velocity > 0:
                note = msg.note
                note_counts[note] += 1
                total_notes += 1
                note_sequence.append(note)

    # Compute transitions for each k separately
    for i in range(len(note_sequence)):
        for k in range(1, max_k + 1):
            if i + k < len(note_sequence):
                transition = (note_sequence[i], note_sequence[i + k])
                transition_counts[k][transition] += 1

    # Convert counts to probability distributions
    note_distribution = {note: count / total_notes for note, count in note_counts.items()}

    transition_distributions = {}
    for k in range(1, max_k + 1):
        total_transitions = sum(transition_counts[k].values())
        transition_distributions[k] = {pair: count / total_transitions for pair, count in transition_counts[k].items()}

    return note_distribution, transition_distributions


# Generate an initial sequence
def generate_initial_sequence(target_distribution, length):
    """
    Generates an initial sequence of notes based on the target note distribution.
    """
    notes = list(target_distribution.keys())
    probabilities = list(target_distribution.values())
    return [np.random.choice(notes, p=probabilities) for _ in range(length)]


# Compute note and transition distributions from a sequence
def compute_sequence_statistics(sequence, max_k=10):
    """
    Computes:
    - note_distribution: {note -> probability}
    - transition_distributions: {k -> {(note1, note2) -> probability}}
    """
    note_counts = defaultdict(int)
    transition_counts = {k: defaultdict(int) for k in range(1, max_k + 1)}
    total_notes = len(sequence)

    for i in range(total_notes):
        note_counts[sequence[i]] += 1
        for k in range(1, max_k + 1):
            if i + k < total_notes:
                transition = (sequence[i], sequence[i + k])
                transition_counts[k][transition] += 1

    note_distribution = {note: count / total_notes for note, count in note_counts.items()}

    transition_distributions = {}
    for k in range(1, max_k + 1):
        total_transitions = sum(transition_counts[k].values())
        transition_distributions[k] = {pair: count / total_transitions for pair, count in transition_counts[k].items()}

    return note_distribution, transition_distributions


# Compute KL divergence for notes and all transition distributions
def total_kl_divergence(seq_distribution, target_distribution, weight=0.5, max_k=10):
    """
    Computes a weighted sum of KL divergences:
    - `KL(notes_generated || notes_target)`
    - `Σ KL(transitions_generated[k] || transitions_target[k])` for k in {1, ..., max_k}
    """
    all_notes = set(target_distribution[0].keys()).union(seq_distribution[0].keys())
    p_notes = np.array([seq_distribution[0].get(note, 1e-9) for note in all_notes])
    q_notes = np.array([target_distribution[0].get(note, 1e-9) for note in all_notes])
    kl_notes = entropy(p_notes, q_notes)

    kl_transitions_sum = 0
    for k in range(1, max_k + 1):
        all_transitions = set(target_distribution[1][k].keys()).union(seq_distribution[1][k].keys())
        p_transitions = np.array([seq_distribution[1][k].get(pair, 1e-9) for pair in all_transitions])
        q_transitions = np.array([target_distribution[1][k].get(pair, 1e-9) for pair in all_transitions])
        kl_transitions_sum += entropy(p_transitions, q_transitions)

    return weight * kl_notes + (1 - weight) * (kl_transitions_sum / max_k)


# Local search optimization
def optimize_sequence(target_distribution, initial_sequence, max_k=10, weight=0.5):
    """
    Optimizes the sequence to match the target distributions using local search.
    """
    current_sequence = initial_sequence[:]
    current_stats = compute_sequence_statistics(current_sequence, max_k)
    current_kl = total_kl_divergence(current_stats, target_distribution, weight, max_k)

    notes = list(target_distribution[0].keys())

    for it in range(max_iterations):
        # Choose a random position and replace with another note
        new_sequence = current_sequence[:]
        idx = random.randint(0, len(new_sequence) - 1)
        new_note = random.choice(notes)
        new_sequence[idx] = new_note

        # Compute new KL divergence
        new_stats = compute_sequence_statistics(new_sequence, max_k)
        new_kl = total_kl_divergence(new_stats, target_distribution, weight, max_k)

        # Accept change if it improves KL divergence
        if new_kl < current_kl:
            current_sequence = new_sequence
            current_kl = new_kl
            print(f"{it}:{new_kl}")

    return current_sequence


# Save sequence to MIDI file
def save_midi(sequence, output_file, tempo=500000):
    """
    Saves a sequence of notes as a MIDI file.
    """
    mid = mido.MidiFile()
    track = mido.MidiTrack()
    mid.tracks.append(track)

    track.append(mido.MetaMessage('set_tempo', tempo=tempo))

    for note in sequence:
        track.append(mido.Message('note_on', note=note, velocity=64, time=0))
        track.append(mido.Message('note_off', note=note, velocity=64, time=480))

    mid.save(output_file)


# Main function
def generate_midi_imitation(input_midi, output_midi, sequence_length=100, max_k=10, weight=0.5):
    """
    Generates a MIDI sequence imitating the style of the input MIDI.
    """
    print("Extracting note and transition distributions from target MIDI...")
    target_distribution = extract_midi_statistics(input_midi, max_k)

    print("Generating initial random sequence...")
    initial_sequence = generate_initial_sequence(target_distribution[0], sequence_length)

    print("Optimizing sequence to match note and transition distributions...")
    optimized_sequence = optimize_sequence(target_distribution, initial_sequence, max_k, weight=weight)

    print(f"Saving generated MIDI to {output_midi}...")
    save_midi(optimized_sequence, output_midi)

    print("Done!")


# Example usage:
# generate_midi_imitation("target.mid", "output.mid", sequence_length=200, max_k=10, weight=0.5)

# Example usage:
max_iterations = 20000
generate_midi_imitation("../data/bach_partita_mono.mid", "output.mid", sequence_length=30, max_k=7, weight=0.5)
# generate_midi_imitation("../data/prelude_c.mid", "output.mid", sequence_length=30, max_k=6, weight=0.5)
