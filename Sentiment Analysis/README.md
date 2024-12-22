# Sentiment Analysis on IMDB Reviews

This project performs sentiment analysis on IMDB movie reviews, classifying them as either positive or negative using a neural network model built with TensorFlow and Keras.

### Requirements
1. Python 3.x
2. TensorFlow
3. Pandas
4. NumPy

### Install dependencies:

`pip install tensorflow pandas numpy`

### Files
IMDB_Dataset.csv: Contains IMDB movie reviews with sentiment labels (positive/negative). (Note: This file is too large to upload to GitHub)
word_index.csv: Maps words to integer indices for encoding the text data.

### Dataset
The dataset used in this project can be downloaded from "https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews".

### Overview
1. Data Preparation: The dataset is loaded, and reviews are preprocessed by lowering case, removing punctuation, and tokenizing into words.
2. Word Indexing: Reviews are encoded into integer sequences using a predefined word index, with special tokens for padding and unknown words.
3. Model Architecture: A neural network with an embedding layer, a global average pooling layer, and dense layers is used to classify sentiment.
4. Training & Evaluation: The model is trained on the dataset for 30 epochs, evaluated on test data, and used for sentiment prediction.

### Example Usage
The model can be used to predict the sentiment of a random review from the test set. After training, it evaluates a review and prints whether the sentiment is positive or negative.
