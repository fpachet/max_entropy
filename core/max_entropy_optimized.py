import time

import numpy as np
import mido
from collections import Counter
from scipy.optimize import minimize
import random
from line_profiler_pycharm import profile


class MaxEntropyMelodyGenerator:
    def __init__(self, midi_file, Kmax=10, lambda_reg=1.0):
        self.elapsed_ns_in_function= 0
        self.midi_file = midi_file
        self.Kmax = Kmax
        self.lambda_reg = lambda_reg
        self.notes = self.extract_notes()
        for _ in range (Kmax):
            self.notes.insert(0, -1)
            self.notes.append(-2)
        self.note_set = list(set(self.notes))
        self.note_to_idx = {note: i for i, note in enumerate(self.note_set)}
        self.idx_to_note = {i: note for note, i in self.note_to_idx.items()}
        self.padding_start_idx = self.note_to_idx[-1]
        self.padding_stop_idx = self.note_to_idx[-2]
        self.seq = np.array([self.note_to_idx[note] for note in self.notes])
        self.cpt_iterations = 0

    def vocabulary_size(self):
        return len(self.note_set)
    def extract_notes(self):
        """ Extracts MIDI note sequence from a MIDI file. """
        mid = mido.MidiFile(self.midi_file)
        notes = []
        if (len(mid.tracks)) == 1:
            track =  mid.tracks[0]
        else:
            track = mid.tracks[1]
        for msg in track:
            if msg.type == 'note_on' and msg.velocity > 0:
                notes.append(msg.note)
        return notes
    def compute_partition_function_old(self, h, J, context):
        z = 0
        for sigma in range(self.vocabulary_size()):
            energy = h[sigma]
            for k in range(self.Kmax):
                j_k = J[k]
                if -k-1 in context:
                    energy += j_k[context[-k-1], sigma]
                if k+1 in context:
                    energy += j_k[sigma, context[k+1]]
            z += np.exp(energy)
        return z

    import numpy as np
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

    compute_partition_function = compute_partition_function_with_tests
    @profile
    def negative_log_likelihood(self, params):
        self.cpt_iterations = self.cpt_iterations + 1
        h = params[:self.vocabulary_size()]
        J_flat = params[self.vocabulary_size():]
        J = {k: J_flat[k * self.vocabulary_size() ** 2:(k + 1) * self.vocabulary_size() ** 2].reshape(
            (self.vocabulary_size(), self.vocabulary_size()))
            for k in range(self.Kmax)}
        # J2 = np.array(params[self.vocabulary_size():]).reshape((self.Kmax, self.vocabulary_size(), self.vocabulary_size()))
        loss = 0
        M = len(self.seq)
        for i in range(self.Kmax, M - self.Kmax):
            s_0 = self.seq[i]
            context = {k: self.seq[i + k] for k in range(self.Kmax + 1) if i + k < M}
            context.update({-k: self.seq[i - k] for k in range(self.Kmax + 1) if i - k >= 0})
            Z = self.compute_partition_function(h, J, context)
            energy = h[s_0] + self.sum_energy_in_context(J, context, s_0)
            energy2 = h[s_0] + self.sum_energy_in_context_old(J, context, s_0)
            if energy != energy2:
                print("erreur energy")
            energy -= np.log(Z)
            loss += energy
        loss *= -1 / M
        l1_reg = sum(np.abs(J[k]).sum() for k in range(self.Kmax))
        loss += (self.lambda_reg / M) * l1_reg
        print(loss)
        return loss

    @profile
    def sum_energy_in_context_old(self, J, context, s_0):
        energy = 0
        for k in range(self.Kmax):
           j_k = J[k]
           if -k-1 in context:
               energy += j_k[context.get(-k - 1), s_0]
           if k+1 in context:
               energy += j_k[s_0, context.get(k + 1)]
        return energy

    def sum_energy_in_context(self, J, context, s_0):
        energy = 0

        # Iterate over k values while checking context keys efficiently
        for k in range(self.Kmax):
            j_k = J[k]  # Access J[k] directly
            neg_idx, pos_idx = -k - 1, k + 1
            # Check and accumulate energy from valid context indices
            if neg_idx in context:
                energy += j_k[context[neg_idx], s_0]
            if pos_idx in context:
                energy += j_k[s_0, context[pos_idx]]
        return energy

    @profile
    def gradient(self, params):
        h = params[:self.vocabulary_size()]
        J_flat = params[self.vocabulary_size():]
        voc_size_2 = self.vocabulary_size() ** 2
        J = {k: J_flat[k * voc_size_2:(k + 1) * voc_size_2].reshape(
            self.vocabulary_size(), self.vocabulary_size())
            for k in range(self.Kmax)}
        grad_h = np.zeros_like(h)
        grad_J = {k: np.zeros_like(J[k]) for k in range(self.Kmax)}
        M = len(self.seq)
# local fields
        for r in range(self.vocabulary_size()):
            # if r == self.padding_idx:
            #     continue
            grad_h_r = 0
            for i in range(M):
                s_0 = self.seq[i]
                context = {k: self.seq[i + k] for k in range(self.Kmax + 1) if i + k < M}
                context.update({-k: self.seq[i - k] for k in range(self.Kmax + 1) if i - k >= 0})
                Z = self.compute_partition_function(h, J, context)
                if r == s_0:
                    grad_h_r += 1
                energy = np.exp(h[r] + self.sum_energy_in_context(J, context, r))
                energy2 = np.exp(h[r] + self.sum_energy_in_context_old(J, context, r))
                if energy != energy2:
                    print("error energy")
                expo = energy
                grad_h_r -= expo / Z
            grad_h[r] = -grad_h_r / M
# J
        for k in range(self.Kmax):
            for r in range(self.vocabulary_size()):
                for r2 in range(self.vocabulary_size()):
                    for i in range(M):
                        s_0 = self.seq[i]
                        context = {l: self.seq[i + l] for l in range(self.Kmax + 1) if i + l < M}
                        context.update({-l: self.seq[i - l] for l in range(self.Kmax + 1) if i - l >= 0})
                        Z = self.compute_partition_function(h, J, context)
                        prob = 0
                        if i - k - 1 >= 0 and r == self.seq[i - k - 1] and r2 == s_0:
                            prob += 1
                        if i + k + 1 < M and r == s_0 and r2 == self.seq[i + k + 1]:
                            prob +=  1
                        if i - k - 1 >= 0 and r == self.seq[i - k - 1]:
                            prob -= np.exp(h[r2] + self.sum_energy_in_context(J, context, r2))
                        if i + k + 1 < M and r2 == self.seq[i + k + 1]:
                            prob -= np.exp(h[r] + self.sum_energy_in_context(J, context, r))
                    grad_J[k][r][r2] = -prob / M
                    grad_J[k][r][r2] += (self.lambda_reg / M) * np.abs(J[k][r][r2])
        grad_J_flat = np.concatenate([grad_J[k].flatten() for k in range(self.Kmax)])
        return np.concatenate([grad_h, grad_J_flat])

    def train(self, max_iter = 1000):
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
            proposed_note = self.note_to_idx[random.choice([elt for elt in self.note_set if elt != current_note])]
            context = {k: sequence[idx + k] for k in range(self.Kmax + 1) if idx + k < length}
            context.update({-k: sequence[idx - k] for k in range(self.Kmax + 1) if idx - k >= 0})
            current_energy = h[current_note] + self.sum_energy_in_context(J, context, current_note)
            proposed_energy = h[proposed_note] + self.sum_energy_in_context(J, context, proposed_note)
            acceptance_ratio = min(1, np.exp(proposed_energy - current_energy))
            if random.random() < acceptance_ratio:
                # print(f"change {current_note} to {proposed_note}")
                sequence[idx] = proposed_note
        # build sequence of notes
        result = [self.idx_to_note[i] for i in sequence]
        return result
    def save_midi(self, sequence, output_file="generated.mid"):
        mid = mido.MidiFile()
        track = mido.MidiTrack()
        mid.tracks.append(track)
        for note in sequence:
            if note == self.padding_start_idx or note == self.padding_stop_idx:
                print("silence found")
                continue
            track.append(mido.Message('note_on', note=note, velocity=64, time=200))
            track.append(mido.Message('note_off', note=note, velocity=64, time=200))
        mid.save(output_file)

# Utilisation
generator = MaxEntropyMelodyGenerator("../data/test_sequence_2notes.mid", Kmax=3)
# generator = MaxEntropyMelodyGenerator("../data/test_sequence_arpeggios.mid", Kmax=6)
t0 = time.perf_counter_ns()
h_opt, J_opt = generator.train(max_iter=4)
t1 = time.perf_counter_ns()
print(f"time: {(t1-t0)/1000000}")
print(f"time: {generator.elapsed_ns_in_function/1000000}ms")
generated_sequence = generator.generate_sequence_metropolis(h_opt, J_opt, burn_in=2000, length=50)
generator.save_midi(generated_sequence, "generated_melody.mid")
