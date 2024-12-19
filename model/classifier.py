import numpy as np
import pickle
import csv

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
# load pre-trained embeddings
embeddings = np.load("embeddings.npy") 

# Load voc-to-idx mapping used during training
with open("vocab.pkl", "rb") as f:
    vocab = pickle.load(f)

def tweet_to_embedding(tweet, embeddings, vocab):
    # Convert tweet to a list of word embeddings
    words = tweet.split()
    word_vecs = [embeddings[vocab[word]] for word in words if word in vocab]
    if len(word_vecs) == 0:
        return np.zeros(embeddings.shape[1])  # Default to zero vector if no valid words
    return np.mean(word_vecs, axis=0)

# Load training data from files
def load_training_data(pos_file, neg_file):
    """
    Load tweets and labels from positive and negative tweet files.
    """
    with open(pos_file, "r", encoding="utf-8") as f:
        pos_tweets = f.readlines()
    with open(neg_file, "r", encoding="utf-8") as f:
        neg_tweets = f.readlines()

    tweets = pos_tweets + neg_tweets
    labels = [1] * len(pos_tweets) + [0] * len(neg_tweets)

    return tweets, labels

# Load training data
tweets, labels = load_training_data("data/train_pos.txt", "data/train_neg.txt")

# Convert tweets to features
X = np.array([tweet_to_embedding(tweet.strip(), embeddings, vocab) for tweet in tweets])
Y = np.array(labels)

# Split data into training and validation sets
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=44)

# Train a simple logistic regression classifier
classifier = LogisticRegression(max_iter=100, verbose=3)
classifier.fit(X_train, Y_train)

y_val_pred = classifier.predict(X_val)
print(f"Validation accuracy: {accuracy_score(Y_val, y_val_pred)}")

# Load test data and evaluate
with open("data/test_data.txt", "r", encoding="utf-8") as f:
    test_tweets = f.readlines()

X_test = np.array([tweet_to_embedding(tweet.strip(), embeddings, vocab) for tweet in test_tweets])
test_predictions = classifier.predict(X_test)

# Generate submission file
with open("submission.csv", "w", newline="") as f:
    writer = csv.writer(f)
    # Write header
    writer.writerow(["Id", "Prediction"])
    # Write predictions, with 1-based indexing for Id and -1 for negative, 1 for positive
    for i, pred in enumerate(test_predictions, start=1):
        writer.writerow([i, 1 if pred == 1 else -1])
