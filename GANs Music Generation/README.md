# Music Generation with Transformer-Based GANs

This project implements a Transformer-based Generative Adversarial Network (GAN) for multi-track music generation using the **JSB Chorales Dataset**. It combines advanced deep learning techniques, including Transformers and GANs, to generate expressive and coherent musical sequences. 

<img width="764" alt="Screenshot 2024-12-22 at 10 00 13 PM" src="https://github.com/user-attachments/assets/e09f74e8-a684-4c8d-a4b9-e9d5881b077a" />


## Features
- **Data Preprocessing**: Handles padding and truncation of variable-length sequences for model training.
- **Transformer-Based Models**:
  - **Generator**: Produces multi-track music sequences.
  - **Discriminator**: Evaluates generated sequences as real or fake.
- **Positional Encoding**: Captures sequential information for the Transformer architecture.
- **Custom MIDI Generation**: Converts model outputs into playable MIDI files.
- **Loss Functions**:
  - Generator: Encourages realistic sequence generation.
  - Discriminator: Distinguishes real sequences from generated ones.
- **Training Loop**: Iteratively trains the generator and discriminator while saving generated MIDI outputs.

## Dataset
The **JSB Chorales Dataset** is used and split into training, validation, and test sets. Sequence lengths vary and are preprocessed for consistent input to the model.

## Dependencies
- `numpy`
- `torch`
- `music21`

## Usage
1. Preprocess the data:
   - Pad or truncate sequences to the desired length.
2. Initialize the **Transformer Generator** and **Discriminator**:
   - Configure model parameters such as `d_model`, `num_heads`, and `num_layers`.
3. Train the models:
   - Use the provided training loop with a defined number of epochs.
4. Generate MIDI files:
   - Use the `generate_midi_from_sequence` function to produce MIDI outputs from the trained generator.

## Model Highlights
- **Generator**: Utilizes a Transformer with multi-head attention to generate coherent sequences across tracks.
- **Discriminator**: Validates sequences using similar Transformer-based architecture with a binary output.
- **Positional Encoding**: Enhances temporal awareness of the models, which is crucial for sequential data like music.

## Output
- Trained models generate multi-track MIDI files with expressive note dynamics and structure.

## Future Work
- Fine-tuning hyperparameters for improved generation quality.
- Extending the architecture for polyphonic and higher-dimensional music datasets.
- Incorporating additional evaluation metrics for generated music quality.
