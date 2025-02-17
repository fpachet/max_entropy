import numpy as np
import mido
from scipy.optimize import minimize


class MaxEntropyMusic:
    def __init__(self, k_max=5, mc_steps=10000):
        self.k_max = k_max  # Maximum interaction distance
        self.mc_steps = mc_steps  # Monte Carlo sampling steps
        self.h = None  # Local fields
        self.J = None  # Interaction potentials

    def preprocess_midi(self, midi_file):
        """Load MIDI file and extract note sequences."""
        mid = mido.MidiFile(midi_file)
        notes = []
        for msg in mid.tracks[0]:
            if msg.type == 'note_on' and msg.velocity > 0:
                notes.append(msg.note)
        return np.array(notes)

    def max_entropy_objective(self, params, f_target, fk_target, unique_notes):
        """Objective function for maximum entropy optimization."""
        q = len(unique_notes)
        h = dict(zip(unique_notes, params[:q]))
        J = {}
        index = q
        for k in range(1, self.k_max + 1):
            J[k] = {}
            for pair in fk_target[k]:
                J[k][pair] = params[index]
                index += 1

        # Compute model frequencies
        f_model = {note: np.exp(h[note]) for note in unique_notes}
        Z = sum(f_model.values())
        for note in f_model:
            f_model[note] /= Z

        fk_model = {}
        for k in range(1, self.k_max + 1):
            fk_model[k] = {}
            for pair in fk_target[k]:
                fk_model[k][pair] = np.exp(J[k][pair]) / Z

        # Compute loss
        loss = sum((f_model[note] - f_target[note]) ** 2 for note in unique_notes)
        for k in range(1, self.k_max + 1):
            for pair in fk_target[k]:
                loss += (fk_model[k][pair] - fk_target[k][pair]) ** 2

        return loss
    def fit(self, sequence):
        """Estimate the model parameters using Maximum Likelihood Estimation."""
        f_target, fk_target = self.compute_frequencies(sequence)
        unique_notes = list(f_target.keys())
        num_params = len(unique_notes) + sum(len(fk_target[k]) for k in range(1, self.k_max + 1))

        initial_params = np.random.randn(num_params) * 0.01
        result = minimize(self.max_entropy_objective, initial_params, args=(f_target, fk_target, unique_notes),
                          method='L-BFGS-B')
        # Extract optimized parameters
        q = len(unique_notes)
        self.h = dict(zip(unique_notes, result.x[:q]))
        self.J = {}
        index = q
        for k in range(1, self.k_max + 1):
            self.J[k] = {}
            for pair in fk_target[k]:
                self.J[k][pair] = result.x[index]
                index += 1
        print(f_target)
        print(self.h)
        print(fk_target)
        print(self.J)
    def compute_frequencies(self, sequence):
        """Compute empirical single and pairwise frequencies."""
        unique_notes = np.unique(sequence)
        q = len(unique_notes)
        f = {note: np.mean(sequence == note) for note in unique_notes}
        fk = {}

        for k in range(1, self.k_max + 1):
            fk[k] = {}
            for i in range(len(sequence) - k):
                pair = (sequence[i], sequence[i + k])
                if pair not in fk[k]:
                    fk[k][pair] = 0
                fk[k][pair] += 1
            for pair in fk[k]:
                fk[k][pair] /= (len(sequence) - k)

        return f, fk

    def metropolis_sampling(self, length=100):
        """Generate a new sequence using Metropolis-Hastings sampling with full sequence context."""
        if self.h is None or self.J is None:
            raise ValueError("Model parameters are not trained yet.")

        unique_notes = list(self.h.keys())
        sequence = [np.random.choice(unique_notes)]

        for _ in range(1, length):
            current_note = sequence[-1]
            candidate_note = np.random.choice(unique_notes)

            # Compute energy difference considering full sequence history
            delta_energy = self.h.get(candidate_note, 0) - self.h.get(current_note, 0)
            for k in range(1, min(self.k_max + 1, len(sequence) + 1)):
                delta_energy += self.J.get(k, {}).get((sequence[-k], candidate_note), 0) - \
                                self.J.get(k, {}).get((sequence[-k], current_note), 0)

            acceptance_prob = min(1, np.exp(-delta_energy))
            if np.random.rand() < acceptance_prob:
                sequence.append(candidate_note)
            else:
                sequence.append(current_note)

        return sequence

    def save_to_midi(self, sequence, filename="generated.mid"):
        """Save the generated sequence as a MIDI file with longer notes and faster tempo."""
        mid = mido.MidiFile()
        track = mido.MidiTrack()
        mid.tracks.append(track)
        for note in sequence:
            track.append(mido.Message('note_on', note=note, velocity=64, time=80))  # Faster tempo
            track.append(mido.Message('note_off', note=note, velocity=64, time=300))  # Longer notes
        mid.save(filename)
        print(f"MIDI file saved as {filename}")


# Example Usage:
music_model = MaxEntropyMusic(k_max=4, mc_steps=1000)
sequence_path = '../../data/test_sequence_2notes.mid'
sequence = music_model.preprocess_midi(sequence_path)
music_model.fit(sequence)
generated_sequence = music_model.metropolis_sampling(10)
music_model.save_to_midi(generated_sequence, "output.mid")
print("Generated MIDI Notes:", generated_sequence)
