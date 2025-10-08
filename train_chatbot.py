import json
import random
import nltk
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Download NLTK data
nltk.download('punkt')

# Load the intents file
with open('intents.json') as file:
    data = json.load(file)

# Prepare training data
sentences = []
labels = []
for intent in data['intents']:
    for pattern in intent['patterns']:
        sentences.append(pattern)
        labels.append(intent['tag'])

# Convert text to features
vectorizer = CountVectorizer(tokenizer=nltk.word_tokenize)
X = vectorizer.fit_transform(sentences)
y = np.array(labels)

# Train a simple classifier
model = MultinomialNB()
model.fit(X, y)

# Save model components
import pickle
with open("chatbot_model.pkl", "wb") as f:
    pickle.dump((model, vectorizer, data), f)

print("✅ Model trained and saved successfully!")
