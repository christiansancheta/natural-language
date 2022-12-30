import json
import nltk
from nltk.classify import NaiveBayesClassifier

with open('data.json', 'r') as f:
    data = json.load(f)

train_data = [(entry['text'], entry['label']) for entry in data]

def extract_features(document):
    document_words = set(document)
    features = {}
    for word in document_words:
        features[word] = (word in document_words)
    return features

classifier = NaiveBayesClassifier.train(train_data)

test_data = ...
accuracy = nltk.classify.accuracy(classifier, test_data)
print("Accuracy:", accuracy)
