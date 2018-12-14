#Imports
import re
import guidedlda
import numpy as np
import pandas as pd

import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel

#Custom Imports
from functions import *

#Import Data
tickets_excel = 'tickets-con.xls'
tickets = pd.read_excel(tickets_excel, usecols=[0,1,2])

#print(tickets.head(10))
#PreProcess Data
data = tickets.content.values.tolist()
data = [re.sub('\S*@\S*\s?', '', str(sent)) for sent in data]
data = [re.sub('\s+',' ',sent) for sent in data]
data = [re.sub('\'',"",sent) for sent in data]

#Tokenize words
data_words = list(sent_to_words(data))

#Bigram and Trigram models
bigram = gensim.models.Phrases(data_words,min_count=5,threshold=100)
trigram = gensim.models.Phrases(bigram[data_words],threshold=100)
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)

data_words_nostop = remove_stopwords(data_words)
data_words_bigrams = make_bigrams(data_words_nostop,bigram_mod)

data_lemzed = lemmatization(data_words_bigrams, allowed=['NOUN','ADJ','VERB','ADV'])

id2word = corpora.Dictionary(data_lemzed)
corpus = [id2word.doc2bow(text) for text in data_lemzed]

model = guidedlda.GuidedLDA(n_topics=5, n_iter=100, random_state=7, refresh=20)
topic_word = model.topic_word_
n_top_words = 8
for i,topic_dist in enumerate(topic_word):
    topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
    print('Topic {}: {}'.format(i, ''.join(topic_words)))
