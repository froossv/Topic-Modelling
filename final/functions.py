import gensim
import spacy
import pandas as pd
from gensim.utils import simple_preprocess
from nltk.corpus import stopwords
import custom_stopwords
stop_words = stopwords.words('english')
stop_words.extend(['from','subject','re','edu','use'])
nlp = spacy.load('en',disable = ['parser','ner'])

def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc =True))

def remove_stopwords(texts):
    extend_stop_words()
    #print(stop_words)
    #input("Waiting")
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
    df = pd.read_json('en.json')
    words = df.values.tolist()
    stop_words.extend(i for i in words)

    words = custom_stopwords.words
    stop_words.extend(i for i in words)

def format_topics_sentences(ldamodel,corpus, texts):
    sent_topics_df = pd.DataFrame()

    for i, row in enumerate(ldamodel[corpus]):
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word,prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic','Perc_Contribution','Topic_Keywords']
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return(sent_topics_df)
