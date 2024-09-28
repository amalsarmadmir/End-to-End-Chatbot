
import random
from .model import get_model
from .intents import intents

# Load the trained model and vectorizer
vectorizer, clf = get_model()

def chatbot_response(input_text):
    # Transform the input to the model's format and predict the intent
    input_text = vectorizer.transform([input_text])
    tag = clf.predict(input_text)[0]

    # Find the matching response
    for intent in intents:
        if intent['tag'] == tag:
            response = random.choice(intent['responses'])
            return response