import torch
import torch.nn as nn
import torch.optim as optim
import pretty_midi
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2LMHeadModel, GPT2Config


# Load MIDI file and extract notes
def midi_to_notes(midi_path):
    midi_data = pretty_midi.PrettyMIDI(midi_path)
    notes = []
    for instrument in midi_data.instruments:
        for note in instrument.notes:
            notes.append(note.pitch)
    return notes


# Convert sequence of notes into input-output pairs
def create_sequences(notes, seq_length=10):  # Reduced sequence length
    sequences = []
    next_notes = []
    for i in range(len(notes) - seq_length):
        sequences.append(notes[i:i + seq_length])
        next_notes.append(notes[i + seq_length])
    return np.array(sequences), np.array(next_notes)


# Custom dataset for training
class NotesDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = torch.tensor(sequences, dtype=torch.long)
        self.targets = torch.tensor(targets, dtype=torch.long)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]


# Define Transformer-based model
def create_model(vocab_size, embed_dim=64, num_heads=2, num_layers=1):  # Reduced model size
    config = GPT2Config(
        vocab_size=vocab_size,
        n_embd=embed_dim,
        n_head=num_heads,
        n_layer=num_layers,
        bos_token_id=0,
        eos_token_id=vocab_size - 1,
        pad_token_id=vocab_size - 1,
    )
    return GPT2LMHeadModel(config)


# Train the model
def train_model(model, dataloader, epochs=5, lr=0.005):  # Fewer epochs, higher learning rate
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            x, y = batch
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x).logits[:, -1, :]
            loss = loss_fn(outputs, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(dataloader)}")


# Generate new MIDI sequence
def generate_notes(model, seed_sequence, length=50):  # Reduced generated length
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    generated = list(seed_sequence)
    input_seq = torch.tensor(seed_sequence, dtype=torch.long).unsqueeze(0).to(device)

    with torch.no_grad():
        for _ in range(length):
            output = model(input_seq).logits[:, -1, :]
            next_note = torch.argmax(output, dim=-1).item()
            generated.append(next_note)
            input_seq = torch.cat((input_seq[:, 1:], torch.tensor([[next_note]], dtype=torch.long).to(device)), dim=1)

    return generated


# Convert note sequence to MIDI
def notes_to_midi(notes, output_path):
    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)
    start_time = 0
    duration = 0.5  # Fixed note duration
    for note in notes:
        midi_note = pretty_midi.Note(velocity=100, pitch=note, start=start_time, end=start_time + duration)
        instrument.notes.append(midi_note)
        start_time += duration
    midi.instruments.append(instrument)
    midi.write(output_path)


# Main script
def main(midi_file, output_midi, seq_length=10, generate_length=50, batch_size=16, epochs=1000):  # Adjusted parameters
    notes = midi_to_notes(midi_file)
    unique_notes = list(set(notes))
    note_to_idx = {note: i for i, note in enumerate(unique_notes)}
    idx_to_note = {i: note for note, i in note_to_idx.items()}

    encoded_notes = [note_to_idx[note] for note in notes]
    sequences, targets = create_sequences(encoded_notes, seq_length)
    dataset = NotesDataset(sequences, targets)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = create_model(vocab_size=len(unique_notes))
    train_model(model, dataloader, epochs=epochs)

    seed_sequence = encoded_notes[:seq_length]
    generated_notes = generate_notes(model, seed_sequence, length=generate_length)
    decoded_notes = [idx_to_note[idx] for idx in generated_notes]

    notes_to_midi(decoded_notes, output_midi)
    print(f"Generated MIDI saved to {output_midi}")


# Example usage
# main('input.mid', 'output.mid')

# Example usage
main('../data/prelude_c.mid', 'output.mid')
