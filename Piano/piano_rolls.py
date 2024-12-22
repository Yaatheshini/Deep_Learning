import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib
matplotlib.use('TkAgg')

logging.basicConfig(level=logging.INFO)

def csv_to_list(csv_file):
    df = pd.read_csv(csv_file, low_memory=False)
    score = []
    for _, row in df.iterrows():
        start_time = row['start_time']
        duration = row['duration']
        pitch = row['pitch']
        velocity = 1  # Default velocity
        label = row['instrument'] if pd.notna(row['instrument']) else 'Unknown'
        score.append([start_time, duration, pitch, velocity, label])
    return score

def position_encoding(score, max_time, max_pitch, d_model=64):
    score_with_encoding = []
    for i, note in enumerate(score):
        start_time, duration, pitch, velocity, label = note
        position = (start_time + 1e-6) / (max_time + 1e-6)  # Added a small value to avoid zero
        positions = np.linspace(0, d_model - 1, d_model)
        sines = np.sin(positions * position * 2 * np.pi / d_model)
        cosines = np.cos(positions * position * 2 * np.pi / d_model)
        position_encoding = np.where(np.arange(d_model) % 2 == 0, sines, cosines)
        note_with_encoding = note + list(position_encoding)
        score_with_encoding.append(note_with_encoding)
        if i % 1000 == 0:
            logging.info(f"Processed {i} notes.")
    return score_with_encoding

def visualize_piano_roll(score, xlabel='Time (seconds)', ylabel='Pitch', colors=['red', 'blue', 'green', 'purple'],
                         figsize=(12, 4), ax=None, dpi=100):
    # Initialize the figure and axis if not provided
    fig = None
    if ax is None:
        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = plt.subplot(1, 1, 1)

    # Extract labels and create a color map
    labels = sorted(set([note[4] for note in score if isinstance(note[4], str)]))
    if not labels:
        logging.warning("No labels found in the dataset. Defaulting to 'Unknown'.")
        labels = ['Unknown']
    color_map = {label: colors[i % len(colors)] for i, label in enumerate(labels)}

    # Adding rectangles for each note
    for note in score:
        start, duration, pitch, velocity, label, *position_encoding = note
        if duration > 0:  # Only plotting notes with valid durations
            rect = patches.Rectangle((start, pitch - 0.5), duration, 1,
                                     linewidth=1, edgecolor='k', facecolor=color_map.get(label, 'gray'), alpha=0.7)
            ax.add_patch(rect)

    # Set axis limits and labels
    try:
        x_min = min(note[0] for note in score)
        x_max = max(note[0] + note[1] for note in score)
        y_min = min(note[2] for note in score)
        y_max = max(note[2] for note in score)

        ax.set_xlim([x_min, x_max])
        ax.set_ylim([y_min - 1.5, y_max + 1.5])
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
    except ValueError:
        logging.error("Unable to set axis limits. The score data may be empty.")
        return

    # Added grid and legend
    ax.grid()
    ax.set_axisbelow(True)
    patches_legend = [patches.Patch(color=color_map[label], label=label) for label in labels]
    ax.legend(handles=patches_legend, loc='upper right')

    # Saving the figure 
    if fig is not None:
        plt.tight_layout()
        plt.savefig("piano_roll_debug.png")  
        logging.info("Graph saved as 'piano_roll_debug.png'.")
        plt.show(block=False)

# Full dataset processing
csv_file = "extracted_notes_data.csv"
score = csv_to_list(csv_file)
score = score[:5000]

max_time = max(note[0] + note[1] for note in score)
max_pitch = max(note[2] for note in score)

score_with_encoding = position_encoding(score, max_time, max_pitch)

# Save processed data
base_columns = ['Start Time', 'Duration', 'Pitch', 'Velocity', 'Instrument']
encoding_columns = [f'Encoding_{i+1}' for i in range(64)]  # Adjusted based on d_model size of 64
columns = base_columns + encoding_columns
df = pd.DataFrame(score_with_encoding, columns=columns)
output_csv = "processed_with_encoding.csv"
df.to_csv(output_csv, index=False)
logging.info("Processed data saved to processed_with_encoding.csv.")

# Visualizing limited number of notes
visualize_piano_roll(score_with_encoding[:5000], colors=['red', 'blue', 'green', 'purple'], figsize=(10, 5))