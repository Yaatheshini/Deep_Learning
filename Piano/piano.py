import pretty_midi
import os
import pandas as pd

# Root folder of the MAESTRO dataset
maestro_folder = 'piano_dataset/'  

# Initialized a list to hold all note data
notes_data = []

# Walking through all the files in the dataset directory
for root, dirs, files in os.walk(maestro_folder):
    for file in files:
        if file.endswith('.midi') or file.endswith('.mid'):
            midi_path = os.path.join(root, file)
            
            try:
                # Load the MIDI file
                midi_data = pretty_midi.PrettyMIDI(midi_path)

                # Iterate through each instrument and note in the MIDI file
                for instrument in midi_data.instruments:

                    for note in instrument.notes:
                        # Getting each note details
                        start_time = note.start
                        end_time = note.end
                        duration = end_time - start_time
                        note_name = pretty_midi.note_number_to_name(note.pitch)
                        pitch = note.pitch

                        # Added each note data to the list
                        instrument.name = "piano"
                        notes_data.append({
                            'file': file,
                            'instrument': instrument.name,
                            'start_time': start_time,
                            'end_time': end_time,
                            'duration': duration,
                            'note_name': note_name,
                            'pitch': pitch
                        })

            except Exception as e:
                print(f"Could not process {midi_path}: {e}")

# Converting the list to a DataFrame for easier manipulation and saving
df = pd.DataFrame(notes_data)

# Saving extracted note data to a CSV file
output_csv = 'extracted_notes_data.csv'
df.to_csv(output_csv, index=False)
print(f"Data saved to {output_csv}")