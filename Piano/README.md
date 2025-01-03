# MIDI Data Processing and Visualization Project

This project processes MIDI files from the MAESTRO dataset, explicitly focusing on piano music. It extracts relevant note information and applies position encoding to the data. The processed data is then visualized as a piano roll and saved for further analysis. The project leverages Python libraries such as pretty_midi, pandas, matplotlib, and numpy.
1. Python 3.x
2. pretty_midi
3. numpy
4. pandas
5. matplotlib

![piano_roll_5000_debug](https://github.com/user-attachments/assets/d60f2f73-ecf6-4c88-9d61-b7314229801b)


### Install dependencies:

`pip install pretty_midi numpy pandas matplotlib`

### Files
extracted_notes_data.csv: Contains extracted note information from the MIDI files.
processed_with_encoding.csv: Contains the note data along with position encodings.
piano_roll_debug.png: A saved image of the generated piano roll visualization.

### Dataset
This project uses MIDI files from the MAESTRO dataset, which primarily contains piano music. Downloaded from: 
"https://www.kaggle.com/datasets/kritanjalijain/maestropianomidi".

### Overview
1. MIDI File Parsing: The project processes the MAESTRO dataset, extracting note-level data, including start time, end time, duration and pitch from the MIDI files with a focus on piano music.
2. Position Encoding: A positional encoding mechanism is applied to the notes, using sinusoidal functions to map each note's position in time and pitch to a high-dimensional space.
3. Data Export: The extracted note data and encoded features are saved to CSV files for future use and analysis.
4. Piano Roll Visualization: The project generates a visual representation of the MIDI data in the form of a piano roll, using matplotlib, where each note is displayed as a coloured rectangle based on its start time, duration, and pitch and saved as a PNG image for easy interpretation.

### Example Usage
Simply run the provided script to extract notes from the MIDI files in the dataset, apply position encoding, save the processed data to a CSV, and generate a piano roll visualization.
