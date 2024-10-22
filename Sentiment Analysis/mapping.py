import pandas as pd
from collections import Counter
import string

# Load the dataset
df = pd.read_csv('IMDB_Dataset.csv')  # Adjust the filename as needed

# Preprocess the text
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    return text.split()  # Tokenize into words

# Combine all reviews into a single list
all_words = []
for review in df['review']:  # Assuming the reviews are in a column named 'review'
    all_words.extend(preprocess_text(review))

# Create a vocabulary
word_counts = Counter(all_words)
word_to_index = {word: idx + 1 for idx, (word, _) in enumerate(word_counts.items())}

# Save the index to a CSV file
word_index_df = pd.DataFrame(list(word_to_index.items()), columns=['Word', 'Index'])
word_index_df.to_csv('word_index.csv', index=False)

print("Index file created with", len(word_to_index), "unique words.")
