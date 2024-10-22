import string
import tensorflow as tf
import tensorflow.keras as keras
import pandas as pd
import numpy as np
import mapping

# Load IMDB reviews dataset
imdb_reviews = pd.read_csv("IMDB_Dataset.csv")
test_reviews = pd.read_csv("IMDB_Dataset.csv")

# Display the first few rows of the dataset
imdb_reviews.head()

# Load and prepare word index
word_index = pd.read_csv("word_index.csv")
word_index.head(n=10)
word_index = dict(zip(word_index.Word, word_index.Index))

# Add special tokens to word index
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["UNK"] = 2
word_index["<UNUSED>"] = 3

# Preprocess text by lowering case and removing punctuation
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    return text.split()  # Tokenize into words

# Encode reviews into integer sequences
def review_encoder(text):
    arr = [word_index.get(word, word_index["UNK"]) for word in text]
    return arr

# Split data into training and testing sets
train_data, train_labels = imdb_reviews['review'], imdb_reviews['sentiment']
test_data, test_labels = test_reviews['review'], test_reviews['sentiment']

# Preprocess training and testing reviews
train_data = train_data.apply(lambda review: [word.lower() for word in review])
test_data = test_data.apply(lambda review: [word.lower() for word in review])

# Encode the reviews
train_data = train_data.apply(review_encoder)
test_data = test_data.apply(review_encoder)

# Encode sentiments as binary values
def encode_sentiments(sentiment):
    if sentiment == 'positive':
        return 1
    else:
        return 0
    
train_labels = train_labels.apply(encode_sentiments)
test_labels = test_labels.apply(encode_sentiments)

# Pad sequences to ensure uniform length
train_data = keras.preprocessing.sequence.pad_sequences(train_data, value=word_index["<PAD>"], padding='post', maxlen=500)
test_data = keras.preprocessing.sequence.pad_sequences(test_data, value=word_index["<PAD>"], padding='post', maxlen=500)

# Build the model
model = keras.Sequential([
    keras.layers.Embedding(182000, 16),  # Embedding layer
    keras.layers.GlobalAveragePooling1D(),  # Pooling layer
    keras.layers.Dense(16, activation='relu'),  # Dense layer with ReLU activation
    keras.layers.Dense(1, activation='sigmoid')  # Output layer with sigmoid activation
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(train_data, train_labels, epochs=30, batch_size=512, validation_data=(test_data, test_labels), verbose=0)

# Evaluate the model on test data
loss, accuracy = model.evaluate(test_data, test_labels)

# Display the first few test reviews
test_reviews.head()

# Randomly select a review for sentiment analysis
index = np.random.randint(1, 1000)
user_review = test_reviews.loc[index]
print("Review:", user_review['review'])

# Prepare the user review for prediction
user_review = test_data[index]
user_review = np.array([user_review])

# Make sentiment prediction
prediction = model.predict(user_review)

# Determine sentiment based on prediction
if prediction[0] > 0.5:
    sentiment = "positive"
else:
    sentiment = "negative"

# Print the final sentiment for the specific review
print("Sentiment:", sentiment)
