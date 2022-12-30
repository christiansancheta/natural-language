import gensim
from gensim.parsing.preprocessing import strip_tags, strip_multiple_whitespaces, strip_numeric, remove_stopwords, strip_short, strip_punctuation

def preprocess_text(text):
    text = strip_tags(text)
    text = strip_multiple_whitespaces(text)
    text = strip_numeric(text)
    text = remove_stopwords(text)
    text = strip_short(text)
    text = strip_punctuation(text)
    return text

def classify_sentiment(text):
    text = preprocess_text(text)
    model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
    score = model.score(text.split())
    if score > 0:
        return "Positive"
    elif score < 0:
        return "Negative"
    else:
        return "Neutral"

text = "" # Replace this with the string you want to classify
sentiment = classify_sentiment(text)
print(sentiment)
