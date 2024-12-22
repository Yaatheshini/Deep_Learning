import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import math
from music21 import stream, note, chord

# Load the dataset
data = np.load('Jsb16thSeparated.npz', allow_pickle=True, encoding='latin1')
train_data = data['train']
val_data = data['valid']
test_data = data['test']

# Check the structure of the data
print(f"Train set size: {len(train_data)}")
print(f"Validation set size: {len(val_data)}")
print(f"Test set size: {len(test_data)}")

# Get lengths of each sequence in the training set
sequence_lengths = [len(seq) for seq in train_data]

# Display the lengths and some statistics
print(f"Max sequence length: {max(sequence_lengths)}")
print(f"Min sequence length: {min(sequence_lengths)}")
print(f"Average sequence length: {np.mean(sequence_lengths)}")
print(f"Median sequence length: {np.median(sequence_lengths)}")

def pad_sequences(sequences, max_len, pad_value=0):
    padded_sequences = []
    for seq in sequences:
        seq_len = seq.shape[0]
        if seq_len < max_len:
            # Pad with the pad_value
            pad_width = max_len - seq_len
            padded_seq = np.pad(seq, ((0, pad_width), (0, 0)), mode='constant', constant_values=pad_value)
        else:
            # Truncate if necessary (optional, depends on your requirements)
            padded_seq = seq[:max_len]
        padded_sequences.append(padded_seq)
    return np.array(padded_sequences)

# Find max sequence length in the dataset
max_len = max(seq.shape[0] for seq in train_data)

# Pad train_data
train_data_padded = pad_sequences(train_data, max_len)

# Define Dataset Class with Padding and Truncating
class ChoralesDataset(Dataset):
    def __init__(self, data, seq_len):
        self.data = [torch.tensor(item, dtype=torch.float) for item in data]  # Keep multi-track structure
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sequence = self.data[idx]
        # Pad or truncate sequences
        if sequence.size(0) < self.seq_len:
            padding_size = self.seq_len - sequence.size(0)
            sequence = F.pad(sequence, (0, 0, 0, padding_size))  # Pad along the time axis
        elif sequence.size(0) > self.seq_len:
            sequence = sequence[:self.seq_len]
        return sequence


class TransformerGenerator(nn.Module):
    def __init__(self, num_tracks, seq_len, d_model, num_heads, num_layers):
        super().__init__()
        self.input_embedding = nn.Linear(num_tracks, d_model)  # Map num_tracks to d_model
        self.transformer = nn.Transformer(d_model, num_heads, num_layers, batch_first=True)
        self.fc_out = nn.Linear(d_model, num_tracks)  # Output multi-track

    def create_positional_encoding(self, seq_len, feature_dim):
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, feature_dim, 2).float() * -(math.log(10000.0) / feature_dim))
        pe = torch.zeros(seq_len, feature_dim)
        pe[:, 0::2] = torch.sin(position * div_term)  # even dimensions
        pe[:, 1::2] = torch.cos(position * div_term)  # odd dimensions
        return pe.unsqueeze(0)  # Add batch dimension

    def forward(self, x):
        x = self.input_embedding(x)  # Map input to embedding size
        seq_len = x.size(1)  # Get sequence length from the input
        feature_dim = x.size(2)  # Feature dimension (e.g., 4)

        # Create positional encoding with matching sequence length and feature dimension
        positional_encoding = self.create_positional_encoding(seq_len, feature_dim).to(x.device)

        # Add positional encoding to the input
        x = x + positional_encoding
        x = self.transformer(x, x)
        return self.fc_out(x)  # Return multi-track output


class TransformerDiscriminator(nn.Module):
    def __init__(self, num_tracks, seq_len, d_model, num_heads, num_layers):
        super().__init__()
        self.input_embedding = nn.Linear(num_tracks, d_model)  # Map num_tracks to d_model
        self.transformer = nn.Transformer(d_model, num_heads, num_layers, batch_first=True)
        self.fc_out = nn.Linear(d_model, 1)  # Output: real/fake

    def create_positional_encoding(self, seq_len, feature_dim):
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, feature_dim, 2).float() * -(math.log(10000.0) / feature_dim))
        pe = torch.zeros(seq_len, feature_dim)
        pe[:, 0::2] = torch.sin(position * div_term)  # even dimensions
        pe[:, 1::2] = torch.cos(position * div_term)  # odd dimensions
        return pe.unsqueeze(0)  # Add batch dimension

    def forward(self, x):
        x = self.input_embedding(x)  # Map input to embedding size
        seq_len = x.size(1)  # Get sequence length from the input
        feature_dim = x.size(2)  # Feature dimension (e.g., 4)

        # Create positional encoding with matching sequence length and feature dimension
        positional_encoding = self.create_positional_encoding(seq_len, feature_dim).to(x.device)

        # Add positional encoding to the input
        x = x + positional_encoding
        x = self.transformer(x, x)
        return torch.sigmoid(self.fc_out(x.mean(dim=1)))  # Mean-pooling for sequence


# Function to generate MIDI from the sequence
def generate_midi_from_sequence(generator, batch_size=1, seq_len=32, num_tracks=4, output_filename="generated_song.mid"):
    # Step 1: Generate the sequence using the generator
    generated_sequence = generator(torch.randn(batch_size, seq_len, num_tracks))  # Pass random noise for generation
    
    # Step 2: Check for NaN or Inf values in the generated sequence
    if torch.isnan(generated_sequence).any() or torch.isinf(generated_sequence).any():
        print("Error: NaN or Inf values detected in the generated sequence.")
        return  # Exit if invalid values are detected
    
    # Step 3: Sample from the model's distribution using softmax (instead of argmax)
    probabilities = F.softmax(generated_sequence, dim=-1)  # Get probabilities from model's output
    
    # Check if probabilities contain any invalid values
    if torch.isnan(probabilities).any() or torch.isinf(probabilities).any():
        print("Error: NaN or Inf values detected in the probabilities.")
        return  # Exit if invalid values are detected
    
    # Sample a note
    sampled_sequence = torch.multinomial(probabilities.view(-1, num_tracks), 1).squeeze(1)  # Sample a note

    # Step 4: Convert sampled sequence to MIDI (as before)
    midi_stream = stream.Stream()

    for idx in sampled_sequence.cpu().numpy():  # Convert tensor to numpy for iteration
        # Example: Mapping the sampled note index to MIDI parameters (you can customize this)
        pitch = idx % 128  # Use the note index modulo 128 as pitch (MIDI standard)
        velocity = np.random.randint(50, 100)  # Randomize velocity for expressiveness
        duration = np.random.choice([0.25, 0.5, 1.0])  # Randomize note durations (shorter to make it more lively)
        
        # Create the note with randomized velocity and duration
        note_event = note.Note(pitch, quarterLength=duration)
        note_event.volume.velocity = velocity
        
        # Append the note event to the MIDI stream
        midi_stream.append(note_event)

    # Step 5: Write the MIDI stream to a file
    midi_stream.write('midi', fp=output_filename)

# Initialize Dataset and DataLoader
seq_len = 64
batch_size = 32
train_dataset = ChoralesDataset(train_data, seq_len)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Initialize Models
num_tracks = 4
d_model = 256
num_heads = 8
num_layers = 12

generator = TransformerGenerator(num_tracks, seq_len, d_model, num_heads, num_layers)
discriminator = TransformerDiscriminator(num_tracks, seq_len, d_model, num_heads, num_layers)

# Weight initialization
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

generator.apply(init_weights)
discriminator.apply(init_weights)

# Define Losses
def generator_loss(fake_scores, epsilon=1e-8):
    loss = -torch.mean(torch.log(fake_scores + epsilon))
    return loss

def discriminator_loss(real_scores, fake_scores, epsilon=1e-8):
    
    # Ensure tensors are of the same shape
    assert real_scores.shape == fake_scores.shape, \
        f"Shape mismatch: real_scores {real_scores.shape}, fake_scores {fake_scores.shape}"

    # Apply squeeze to remove any extra dimensions
    real_scores = real_scores.squeeze(-1)
    fake_scores = fake_scores.squeeze(-1)

    # Log-probability loss
    loss = -torch.mean(torch.log(real_scores + epsilon) + torch.log(1 - fake_scores + epsilon))
    return loss

num_epochs = 100

# Example training loop snippet to show loss computation
for epoch in range(num_epochs): 
    for batch in train_loader:
        # Generate fake scores from the generator
        fake_data = generator(batch)
        real_scores = discriminator(batch)
        fake_scores = discriminator(fake_data)

        # Compute losses
        loss_D = discriminator_loss(real_scores, fake_scores)
        loss_G = generator_loss(fake_scores)
        
        # Print loss and gradients for debugging
        print(f"Epoch {epoch}, Loss D: {loss_D.item()}, Loss G: {loss_G.item()}")
    
    # After each epoch, generate a MIDI file from the model
    generate_midi_from_sequence(generator, batch_size=1, seq_len=seq_len, num_tracks=num_tracks, output_filename=f"generated_song_epoch_{epoch}.mid")
