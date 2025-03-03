import numpy as np
import mido
import random
import time
from difflib import SequenceMatcher

"""
Implementation of Continuator more or less as original.
Build a representation for all contexts of size 1 to K and their continuations
Like in the original, continuations are actual realizations in the concrete input sequences.
Contexts are contexts of viewpoints.
Sampling function is quite smart and attempts to avoid too long repetition (max order)
by avoiding singletons when it can
Sequences are considered as monophonic but durations are handled smartly by looking at actual durations in realizations
"""
class Note:
    def __init__(self, pitch, velocity, duration, start_time=0):
        self.pitch = pitch
        self.velocity = velocity
        # the duration in the original sequence
        self.duration = duration
        # the start time in the original sequence
        self.start_time = start_time

    def set_duration(self, d):
        self.duration = d

    def set_start_time(self, t):
        self.start_time = t

    def is_start_note(self):
        return False
    def is_end_note(self):
        return False

    def transpose(self, t):
        return Note(self.pitch + t, self.velocity, self.duration, start_time=self.start_time)

class Start_Note(Note):
    def __init__(self):
        Note.__init__(self, -1, 0, 0)

    def is_start_note(self):
        return True

    def transpose(self, t):
        return self

class End_Note(Note):
    def __init__(self):
        Note.__init__(self, -1, 0, 0)

    def is_end_note(self):
        return True

    def transpose(self, t):
        return self


class Continuator2:

    def __init__(self, midi_file, kmax=5, transposition=False):
        self.input_sequences = []
        self.kmax = kmax
        self.prob_to_keep_singletons = 1 / 3
        self.viewpoints_realizations = {}
        self.terminal_indices = []
        self.prefixes_to_continuations = np.empty(self.kmax, dtype=object)
        for k in range(self.kmax):
            self.prefixes_to_continuations[k] = {}

        self.notes = np.array([], dtype=int)
        # the original sequence
        self.notes_original = self.extract_notes(midi_file)
        # adds start and end notes
        self.notes_original = np.concatenate(([Start_Note()], self.notes_original, [End_Note()]))
        self.input_sequences.append(self.notes_original)
        self.notes = np.concatenate((self.notes, self.notes_original))
        # transpose in 12 tones
        trange = range(0,1)
        if transposition:
            trange = range(-6, 6, 1)
        for t in trange:
            transposed = self.transpose_notes(self.notes_original, t)
            self.notes = np.concatenate((self.notes, transposed))
            self.build_vo_markov_model(transposed)


    def transpose_notes(self, notes, t):
        return [n.transpose(t) for n in notes]

    def is_terminal(self, cont):
        return cont in self.terminal_indices

    def extract_notes(self, midi_file):
        """Extracts the sequence of note-on events from a MIDI file."""
        mid = mido.MidiFile(midi_file)
        notes = []
        pending_notes = np.empty(128, dtype=object)
        pending_start_times = np.zeros(128)
        current_time = 0
        for track in mid.tracks:
            for msg in track:
                current_time += msg.time
                if msg.type == 'note_on' and msg.velocity > 0:
                    new_note = Note(msg.note, msg.velocity, 0)
                    notes.append(new_note)  # Store MIDI note number
                    pending_notes[msg.note] = new_note
                    pending_start_times[msg.note] = current_time
                    new_note.set_start_time(current_time)
                    new_note.set_duration(120)
                if msg.type == 'note_off':
                    pending_note = pending_notes[msg.note]
                    duration = current_time - pending_start_times[msg.note]
                    pending_note.set_duration(duration)
                    pending_notes[msg.note] = None
                    pending_start_times[msg.note] = 0
        return np.array(notes)

    def build_vo_markov_model(self, sequence):
        """Builds a variable-order Markov model for max K order
        accumulates with existing model """

        for k in range(self.kmax):
            prefixes_to_cont_k = self.prefixes_to_continuations[k]
            # TODO indices start at the end of preceding sequences
            for i in range(len(sequence) - k):
                if i < k + 1:
                    continue
                current_ctx = self.get_viewpoint_tuple(tuple(range(i - k - 1, i)))
                if current_ctx not in prefixes_to_cont_k:
                    prefixes_to_cont_k[current_ctx] = []
                prefixes_to_cont_k[current_ctx].append(i)
            self.prefixes_to_continuations[k] = prefixes_to_cont_k
        # rajoute les terminaux
        self.terminal_indices.append(len(sequence) - 1)

    def get_viewpoint(self, index):
        note = self.notes[index]
        vp = tuple([note.pitch, (int)(note.duration / 100)])
        # add the realization to the viewpoint's realizations
        if vp not in self.viewpoints_realizations:
            self.viewpoints_realizations[vp] = []
        if index not in self.viewpoints_realizations[vp]:
            self.viewpoints_realizations[vp].append(index)
        return vp

    def get_viewpoint_tuple(self, indices_tuple):
        vparray = [self.get_viewpoint(id) for id in indices_tuple]
        return tuple(vparray)

    def get_realizations_for_vp(self, vp):

        return self.viewpoints_realizations[vp]

    def random_starting_note(self):
        starting_vp = (-1, 0)
        starting_conts = self.get_realizations_for_vp(starting_vp)
        start = random.choice(starting_conts)
        return start

    def sample_sequence(self, start_note_vp, length=50):
        """Generates a new sequence of notes from the Markov model."""
        # current_seq is a sequence of indices in the original sequence
        starting_conts = self.get_realizations_for_vp(start_note_vp)
        start_note = random.choice(starting_conts)
        current_seq = [start_note]
        if length < 0:
            while True:
                cont = self.get_continuation(current_seq)
                if cont == -1:
                    print("restarting from scratch")
                    cont = self.random_starting_note()
                if self.is_terminal(cont):
                    print("found the end")
                    print(cont)
                    return current_seq
                current_seq.append(cont)
            return current_seq

        for _ in range(length):
            cont = self.get_continuation(current_seq)
            if cont == -1:
                print("restarting from scratch")
                cont = self.random_starting_note()
            current_seq.append(cont)
            print(current_seq)
        return current_seq

    def get_continuation(self, current_seq):
        vp_to_skip = None
        for k in range(self.kmax, 0, -1):
            if k > len(current_seq):
                continue
            continuations_dict = self.prefixes_to_continuations[k - 1]
            # subsequence of indices
            ctx = tuple(current_seq[-k:])
            # get the sequence of viewpoint to lookup the prefix dict
            viewpoint_ctx = self.get_viewpoint_tuple(ctx)
            if viewpoint_ctx in continuations_dict:
                all_conts = continuations_dict[viewpoint_ctx]
                # considers the number of different viewpoints, not the number of continuations
                all_cont_vp = {self.get_viewpoint(i) for i in all_conts}
                if len(all_cont_vp) == 1 and k > 1:
                    # print(f"best continuation is singleton for {k=}: {all_cont_vp}")
                    # proba to skip is proportional to order
                    if random.random() > (1 / (k + 1)):
                        # print(f"skipping continuation for {k=}")
                        vp_to_skip = all_cont_vp.pop()
                        continue
                    else:
                        vp_to_skip = None
                        # print(f"not skipping singleton continuation for {k=}")
                if vp_to_skip is not None and k > 1:
                    all_conts_tu_use = [c for c in all_conts if self.get_viewpoint(c) != vp_to_skip]
                else:
                    all_conts_tu_use = all_conts
                if len(all_conts_tu_use) == 0:
                    print("problem")
                next_continuation = random.choice(all_conts_tu_use)
                # print(
                #     f"found continuation for k {k} with cont size {len(continuations_dict[viewpoint_ctx])} and cont vp size {len(all_cont_vp)}")
                return next_continuation
        return -1
        print("no continuation found")

    def get_pitch_string(self, sequence_of_notes):
        return ''.join([str(note.pitch) + ' ' for note in sequence_of_notes])

    def save_midi(self, idx_sequence, output_file):
        mid = mido.MidiFile()
        track = mido.MidiTrack()
        mid.tracks.append(track)
        # create a new sequence with the right start_times
        sequence = []
        start_time = 0
        for i in idx_sequence:
            note = self.notes[i]
            note_copy = Note(note.pitch, note.velocity, note.duration)
            # keeps the inter note time to be the same as in the original sequence
            if i != 0:
                delta = note.start_time - self.notes[i - 1].start_time
                start_time += delta
            note_copy.set_start_time(start_time)
            sequence.append(note_copy)
        # shift the whole sequence to t=0
        first_note_time = sequence[0].start_time
        for note in sequence:
            note.start_time = note.start_time - first_note_time
        # create all mido messages and sort them
        mido_sequence = []
        for note in sequence:
            try:
                mido_sequence.append(mido.Message('note_on', note=note.pitch, velocity=note.velocity, time=note.start_time))
            except:
                print("Something went wrong")
            mido_sequence.append(
                mido.Message('note_off', note=note.pitch, velocity=0, time=(note.start_time + (int)(note.duration))))
        mido_sequence.sort(key=lambda msg: msg.time)

        current_time = 0
        for msg in mido_sequence:
            delta = msg.time - current_time
            msg.time = delta
            track.append(msg)
            current_time += delta
        mid.save(output_file)
        # plays the file approximately, can be heard of Logic is open
        # with mido.open_output() as output:
        #     for note in sequence:
        #         output.send(mido.Message('note_on', note=note, velocity=64))
        #         time.sleep(0.2)
        #         output.send(mido.Message('note_off', note=note, velocity=64))

    def get_longest_subsequence_with_train(self, sequence_of_idx):
        train_string = generator.get_pitch_string(generator.notes)
        sequence_of_notes = [generator.notes[id] for id in sequence_of_idx]
        sequence_string = generator.get_pitch_string(sequence_of_notes)
        match = SequenceMatcher(None, train_string, sequence_string, autojunk=False).find_longest_match()
        nb_notes_common = train_string[match.a:match.a + match.size].count(' ')
        return nb_notes_common


midi_file_path = "../../data/prelude_c.mid"
# midi_file_path = "../data/bach_partita_mono.midi"
# midi_file_path = "../data/test_sequence_3notes.mid"
t0 = time.perf_counter_ns()
generator = Continuator2(midi_file_path, 4, transposition=False)
t1 = time.perf_counter_ns()
print(f"total time: {(t1 - t0) / 1000000}")
# Sampling a new sequence from the  model
start_note = (-1, 0)  # Pick the INDEX of the first note of the original sequence
generated_sequence = generator.sample_sequence(start_note, length=-1)
generated_sequence = generated_sequence[1:]
generator.save_midi(generated_sequence, "../data/ctor2_output.mid")
print("Generated Sequence:", generated_sequence)

print(f"{generator.get_longest_subsequence_with_train(generated_sequence)} notes in commun with train")
