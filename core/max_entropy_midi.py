import numpy as np
import mido
from scipy.optimize import minimize


class MaxEntropyMusic:
    def __init__(self, k_max=5):
        self.k_max = k_max  # Maximum interaction distance
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

    def generate_sequence(self, length=100):
        """Generate a new sequence using the estimated parameters."""
        if self.h is None or self.J is None:
            raise ValueError("Model parameters are not trained yet.")

        unique_notes = list(self.h.keys())
        sequence = [np.random.choice(unique_notes)]
        for _ in range(1, length):
            probs = {}
            for note in unique_notes:
                energy = self.h[note]
                for k in range(1, self.k_max + 1):
                    if len(sequence) >= k:
                        pair = (sequence[-k], note)
                        if pair in self.J[k]:
                            energy += self.J[k][pair]
                probs[note] = np.exp(energy)

            # Normalize
            total_prob = sum(probs.values())
            for note in probs:
                probs[note] /= total_prob

            # Sample next note
            sequence.append(np.random.choice(unique_notes, p=[probs[n] for n in unique_notes]))

        return sequence

    def save_to_midi(self, sequence, filename="generated.mid"):
        """Save the generated sequence as a MIDI file."""
        mid = mido.MidiFile()
        track = mido.MidiTrack()
        mid.tracks.append(track)
        for note in sequence:
            track.append(mido.Message('note_on', note=note, velocity=64, time=200))
            track.append(mido.Message('note_off', note=note, velocity=64, time=220))
        mid.save(filename)
        print(f"MIDI file saved as {filename}")


# Example Usage:
music_model = MaxEntropyMusic(k_max=10)
print('model done')
sequence_path = '../data/test_sequence_2notes.mid'
# sequence_path = '../data/test_sequence_2notes.mid'
# sequence_path = '../data/bach_partita_mono.mid'
sequence = music_model.preprocess_midi(sequence_path)
print(sequence)
print(len(sequence))
print('preprocess done')
music_model.fit(sequence)
print('fit done')
generated_sequence = music_model.generate_sequence(200)
print('generate done')
music_model.save_to_midi(generated_sequence, "../output.mid")
print("Generated MIDI Notes:", generated_sequence)
print(len(generated_sequence))
