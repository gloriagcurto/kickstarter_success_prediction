#!/usr/bin/python
# -*- coding:utf-8 -*-

"""
Pre-processing of Kickstarter_2020-02-13T03_20_04_893Z.
Text mining. Tokenize, frequent words by category
Input Filename:  data_english.h5
Output Filename: data_frequency_score.h5

Copyright (c) 2020 Gloria G. Curto.
http://gloriagcurto.info
"""
import pandas as pd
import wordcloud
import numpy as np
from nltk.corpus import stopwords
from nltk import FreqDist
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize

# Generate text column
df = pd.read_hdf('../../data/data_english.h5')
df['text'] = df['blurb'] + df['name']
print(df.head())

# Stop_words
stop_words = set(stopwords.words('english'))
stop_words.update([",", "."])


def get_frequency_score(sentence, freqs, stop_words):
    '''
    frequence score 
    '''
    sentence = sentence.replace('?', '').\
                        replace('!', '').\
                        replace(':', '').\
                        replace('"', '').\
                        replace("'", '').\
                        replace('.', '').\
                        replace(',', '').lower()
    
    sentence_words = [word for word in sentence.split() if word not in stop_words]
    # Stemming
    ps = PorterStemmer()
    stemmed_words=[]
    for w in sentence_words:
        stemmed_words.append(ps.stem(w))

    return np.mean([freqs[word] for word in stemmed_words])


# Failed/successful:
f_s = ['successful', 'failed']

#category
cats = df['category_parent_name_ori'].unique()
print(cats)

#wc = wordcloud.WordCloud(stopwords=stop_words, max_words=2000000)
df_freqs = pd.DataFrame()

for cat in cats:
    print(f'Category: {cat}')
    for s in f_s:
        print(f'State: {s}')
        df_sub = df.loc[(df.category_parent_name_ori==cat) & (df.state_grouped==s)]
        text = " ".join([comment for comment in df_sub.text])
        text = text.replace('?', '').\
                        replace('!', '').\
                        replace(':', '').\
                        replace('"', '').\
                        replace("'", '').\
                        replace('.', '').\
                        replace(',', '').lower()
        # lower case, split, and filter stop words
        text_list = [word for word in text.split() if word not in stop_words]
        #text_list = text_list.lower()
        #Stemming
        ps = PorterStemmer()
        stemmed_words_text = [ps.stem(w) for w in text_list]
        
        #Frequency distribution in the category

        frequency_score = []
        freq_dist = FreqDist(stemmed_words_text)
        for row in df_sub['text']:
           frequency_score.append(get_frequency_score(row, freq_dist, stop_words))
                   
        df_sub.loc[:,'frequency_score'] = frequency_score
        df_freqs = pd.concat([df_freqs, df_sub], axis=0)

df_freqs.to_hdf("../../data/data_frequency_score.h5", key="frequency_score")
'''
next = 
id = use as index at the end. Check unique ids/nrow
duplicate data:  df EDA and df timeseries (drop_columns=['category_name', 'category_parent_name',
'location_expanded_country', 'text', 'blurb', 'name','state', 'state_grouped'] in features matrix))
'''