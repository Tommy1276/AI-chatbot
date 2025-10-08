from flask import Flask, render_template, request, jsonify
import pickle
import random

app = Flask(__name__)

# Load the 3 objects from pickle
model, vectorizer, intents_data = pickle.load(open('chatbot_model.pkl', 'rb'))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get", methods=["POST"])
def chatbot_response():
    try:
        user_message = request.json['message']

        # Transform user input
        input_data = vectorizer.transform([user_message])

        # Predict intent
        predicted_tag = model.predict(input_data)[0]

        # Find matching intent and pick a random response
        for intent in intents_data['intents']:
            if intent['tag'] == predicted_tag:
                return jsonify({"response": random.choice(intent['responses'])})

        # Default fallback
        return jsonify({"response": "I'm not sure I understand that. Could you rephrase?"})

    except Exception as e:
        print("Error:", e)
        return jsonify({"response": "Oops! Something went wrong. Please try again."})

if __name__ == "__main__":
    app.run(debug=True)
