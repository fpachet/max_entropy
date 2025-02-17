import time

import numpy as np
import mido
from collections import Counter
from scipy.optimize import minimize
import random
from line_profiler_pycharm import profile


class MaxEntropyMelodyGenerator:
    def __init__(self, midi_file, Kmax=10, lambda_reg=1.0):
        self.elapsed_ns_in_function = 0
        self.midi_file = midi_file
        self.Kmax = Kmax
        self.lambda_reg = lambda_reg
        self.notes = self.extract_notes()
        self.note_set = list(set(self.notes))
        self.note_to_idx = {note: i for i, note in enumerate(self.note_set)}
        self.idx_to_note = {i: note for note, i in self.note_to_idx.items()}
        self.seq = np.array([self.note_to_idx[note] for note in self.notes])
        self.cpt_iterations = 0
        self.cpt_sum_energy = 0
        self.cpt_compute_partition = 0

    def vocabulary_size(self):
        return len(self.note_set)

    def extract_notes(self):
        """ Extracts MIDI note sequence from a MIDI file. """
        mid = mido.MidiFile(self.midi_file)
        notes = []
        if (len(mid.tracks)) == 1:
            track = mid.tracks[0]
        else:
            track = mid.tracks[1]
        for msg in track:
            if msg.type == 'note_on' and msg.velocity > 0:
                notes.append(msg.note)
        return notes

    @profile
    def sum_energy_in_context(self, J, context, center):
        self.cpt_sum_energy += 1
        # center is not necessarily the center of context
        energy = 0
        for k in range(self.Kmax):
            j_k = J[k]
            if -k - 1 in context:
                energy += j_k[context.get(-k - 1), center]
            if k + 1 in context:
                energy += j_k[center, context.get(k + 1)]
        return energy

    def compute_partition_function_old(self, h, J, context):
        self.cpt_compute_partition += 1
        z = 0
        for sigma in range(self.vocabulary_size()):
            energy = h[sigma] + self.sum_energy_in_context(J, context, sigma)
            z += np.exp(energy)
        return z

    @profile
    def compute_partition_function_with_tests(self, h, J, context):
        t0 = time.perf_counter_ns()
        sigma_range = np.arange(self.vocabulary_size())  # Vector of all sigma values
        # Initialize energy with h values
        energy = h[sigma_range]
        # Compute contributions from J matrices efficiently
        for k in range(self.Kmax):
            j_k = J[k]
            if -k - 1 in context:
                energy += j_k[context[-k - 1], sigma_range]  # Broadcasting
            if k + 1 in context:
                energy += j_k[sigma_range, context[k + 1]]  # Broadcasting
        # Compute partition function using vectorized exponentiation and summation
        res = np.sum(np.exp(energy))
        self.elapsed_ns_in_function += time.perf_counter_ns() - t0
        return res

    compute_partition_function = compute_partition_function_old

    @profile
    def negative_log_likelihood(self, params):
        self.cpt_iterations = self.cpt_iterations + 1
        voc_size = self.vocabulary_size()
        h = params[:voc_size]
        J_flat = params[voc_size:]
        J = {k: J_flat[k * voc_size ** 2:(k + 1) * voc_size ** 2].reshape(
            (voc_size, voc_size))
            for k in range(self.Kmax)}
        loss = 0
        M = len(self.seq)
        for i, s_0 in enumerate(self.seq):
            context = self.build_context(self.seq, i)
            Z = self.compute_partition_function(h, J, context)
            energy = h[s_0] + self.sum_energy_in_context(J, context, s_0)
            energy -= np.log(Z)
            loss += energy
        loss *= -1 / M
        l1_reg = sum(np.abs(J[k]).sum() for k in range(self.Kmax))
        loss += (self.lambda_reg / M) * l1_reg
        print(f"{loss=}")
        return loss

    def build_context(self, seq, i):
        M = len(seq)
        context = {k: seq[i + k] for k in range(self.Kmax + 1) if i + k < M}
        context.update({-k: seq[i - k] for k in range(self.Kmax + 1) if i - k >= 0})
        return context


    @profile
    def gradient(self, params):
        voc_size = self.vocabulary_size()
        voc_size_2 = voc_size ** 2
        h = params[:voc_size]
        J_flat = params[voc_size:]
        J = {k: J_flat[k * voc_size_2:(k + 1) * voc_size_2].reshape(
            voc_size, voc_size)
            for k in range(self.Kmax)}
        grad_h = np.zeros_like(h)
        grad_J = {k: np.zeros_like(J[k]) for k in range(self.Kmax)}
        M = len(self.seq)
        # local fields
        for r in range(voc_size):
            sum_grad = 0
            for mu, s_0 in enumerate(self.seq):
                if r == s_0:
                    sum_grad += 1
                context = self.build_context(self.seq, mu)
                Z = self.compute_partition_function(h, J, context)
                expo = np.exp(h[r] + self.sum_energy_in_context(J, context, r))
                sum_grad -= expo / Z
            grad_h[r] = -sum_grad / M
        # J
        for k in range(self.Kmax):
            for r in range(voc_size):
                for r2 in range(voc_size):
                    prob = 0
                    for mu, s_0 in enumerate(self.seq):
                        context = self.build_context(self.seq, mu)
                        Z = self.compute_partition_function(h, J, context)
                        if mu - k - 1 >= 0 and r == self.seq[mu - k - 1]:
                            if r2 == s_0:
                                prob += 1
                            prob -= np.exp(h[r2] + self.sum_energy_in_context(J, context, r2)) / Z
                        if mu + k + 1 < M and r2 == self.seq[mu + k + 1]:
                            if r == s_0:
                                prob += 1
                            prob -= np.exp(h[r] + self.sum_energy_in_context(J, context, r)) / Z
                    grad_J[k][r][r2] = -prob / M + (self.lambda_reg / M) * np.abs(J[k][r][r2])
        grad_J_flat = np.concatenate([grad_J[k].flatten() for k in range(self.Kmax)])
        return np.concatenate([grad_h, grad_J_flat])

    def train(self, max_iter=1000):
        voc_length = self.vocabulary_size()
        voc_2 = voc_length ** 2
        params_init = np.zeros(voc_length + self.Kmax * voc_2)
        res = minimize(self.negative_log_likelihood, params_init, method='L-BFGS-B', jac=self.gradient,
                       options={'maxiter': max_iter})
        return res.x[:voc_length], {
            k: res.x[voc_length + k * voc_2:voc_length + (k + 1) * voc_2].reshape(
                (voc_length, voc_length)) for k in range(self.Kmax)}

    def generate_sequence_metropolis(self, h, J, length=20, burn_in=1000):
        # generate sequence of note indexes
        sequence = [random.choice(self.seq) for _ in range(length)]
        for _ in range(burn_in):
            idx = random.randint(0, length - 1)
            current_note = sequence[idx]
            curent_note_value = self.idx_to_note[current_note]
            proposed_note = self.note_to_idx[random.choice([elt for elt in self.note_set if elt != curent_note_value])]
            context = self.build_context(sequence, idx)
            current_energy = h[current_note] + self.sum_energy_in_context(J, context, current_note)
            proposed_energy = h[proposed_note] + self.sum_energy_in_context(J, context, proposed_note)
            acceptance_ratio = min(1, np.exp(proposed_energy - current_energy))
            if random.random() < acceptance_ratio:
                sequence[idx] = proposed_note
        # build sequence of notes
        result = [self.idx_to_note[i] for i in sequence]
        return result

    def save_midi(self, sequence, output_file="generated.mid"):
        mid = mido.MidiFile()
        track = mido.MidiTrack()
        mid.tracks.append(track)
        for note in sequence:
            track.append(mido.Message('note_on', note=note, velocity=64, time=200))
            track.append(mido.Message('note_off', note=note, velocity=64, time=200))
        mid.save(output_file)


# Utilisation
generator = MaxEntropyMelodyGenerator("../data/test_sequence_3notes.mid", Kmax=2)
# generator = MaxEntropyMelodyGenerator("../data/test_sequence_2notes.mid", Kmax=2)
# generator = MaxEntropyMelodyGenerator("../data/test_sequence_arpeggios.mid", Kmax=6)
t0 = time.perf_counter_ns()
h_opt, J_opt = generator.train(max_iter=6)
print(f"{h_opt=}")
print(f"{J_opt=}")
t1 = time.perf_counter_ns()
print(f"time: {(t1 - t0) / 1000000}")
print(f"time: {generator.elapsed_ns_in_function / 1000000}ms")
print(f"{generator.cpt_sum_energy=}")
print(f"{generator.cpt_compute_partition=}")
print(f"{ len(generator.seq)=}")
generated_sequence = generator.generate_sequence_metropolis(h_opt, J_opt, burn_in=500, length=30)
generator.save_midi(generated_sequence, "../data/generated_melody.mid")
