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

def get_sentiment(text):
    text = preprocess_text(text)
    model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
    sentiment = model.score(text.split())
    return sentiment

text = "" 
sentiment = get_sentiment(text)
print(sentiment)