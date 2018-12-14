#Imports
import re
import pandas as pd

import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel

#Custom Imports
from functions import *
mallet_path = './mallet-2.0.8/bin/mallet'
#Import Data
tickets_excel = 'tickets-con.xls'
tickets = pd.read_excel(tickets_excel, usecols=[0,1,2])

#print(tickets.head(10))
#PreProcess Data
data = tickets.Subject.values.tolist()
data = [re.sub('\S*@\S*\s?', '', str(sent)) for sent in data]
data = [re.sub('\s+',' ',sent) for sent in data]
data = [re.sub('\'',"",sent) for sent in data]

#Tokenize words
data_words = list(sent_to_words(data))

#Bigram and Trigram models
bigram = gensim.models.Phrases(data_words,min_count=5,threshold=100)
bigram_mod = gensim.models.phrases.Phraser(bigram)

data_words_nostop = remove_stopwords(data_words)
data_words_bigrams = make_bigrams(data_words_nostop,bigram_mod)

data_lemzed = lemmatization(data_words_bigrams, allowed=['NOUN','ADJ','VERB','ADV'])

id2word = corpora.Dictionary(data_lemzed)
corpus = [id2word.doc2bow(text) for text in data_lemzed]

#lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,id2word=id2word,num_topics=10,random_state=729,update_every=1,chunksize=100,passes=10,alpha='auto',per_word_topics=True)
#print(lda_model.print_topics())
ldamallet = gensim.models.wrappers.LdaMallet(mallet_path,corpus=corpus,num_topics=10, id2word=id2word)
print(ldamallet.show_topics(formatted=False))
