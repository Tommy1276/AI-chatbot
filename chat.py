import random
import pickle
import nltk

# Load trained model and data
with open("chatbot_model.pkl", "rb") as f:
    model, vectorizer, data = pickle.load(f)

def chatbot_response(text):
    tokens = nltk.word_tokenize(text)
    X = vectorizer.transform([' '.join(tokens)])
    intent = model.predict(X)[0]

    for i in data['intents']:
        if i['tag'] == intent:
            return random.choice(i['responses'])
    return "Sorry, I didn't understand that."

print("Chatbot: Hello! Type 'quit' to exit.")

while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
        print("Chatbot: Goodbye!")
        break
    print("Chatbot:", chatbot_response(user_input))
