import gensim
import spacy
import pandas as pd
from gensim.utils import simple_preprocess
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.extend(['from','subject','re','edu','use'])
nlp = spacy.load('en',disable = ['parser','ner'])



def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc =True))

def remove_stopwords(texts):
    #extend_stop_words()
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

def make_bigrams(texts,phraser):
    return [phraser[doc] for doc in texts]

def lemmatization(texts, allowed=['NOUN','ADJ','VERB','ADV']):
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed])
    return texts_out

def extend_stop_words():
    df = pd.read_csv('male.csv')
    male = df.name.values.tolist()
    stop_words.extend(i for i in male)

    df = pd.read_csv('female.csv')
    female = df.name.values.tolist()
    stop_words.extend(i for i in female)

    df = pd.read_json('en.json')
    words = df.values.tolist()
    stop_words.extend(i for i in words)
