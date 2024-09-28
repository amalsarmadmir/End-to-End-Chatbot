
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from .intents import intents

# Initialize vectorizer and logistic regression model
vectorizer = TfidfVectorizer()
clf = LogisticRegression(random_state=0, max_iter=1000)

# Prepare training data
tags = []
patterns = []
for intent in intents:
    for pattern in intent['patterns']:
        tags.append(intent['tag'])
        patterns.append(pattern)

x = vectorizer.fit_transform(patterns)
y = tags

# Train the model
clf.fit(x, y)

def get_model():
    return vectorizer, clf