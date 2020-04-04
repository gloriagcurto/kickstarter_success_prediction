#!/usr/bin/python
# -*- coding:utf-8 -*-

"""
Pre-processing of Kickstarter_2020-02-13T03_20_04_893Z.
Text mining. Tokenize, filter stop words, stemming. Compute frequent words by category, succesful or failed (remove words in common). Compute score by category (+ freq if word in successful, - frequency if word in failed)
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
from nltk.tokenize import word_tokenize

def df_to_text(df):
    text = " ".join([comment for comment in df.text])
    text = text.replace('?', '').\
                    replace('!', '').\
                    replace('-', ' ').\
                    replace(':', ' ').\
                    replace(';', ' ').\
                    replace('"', '').\
                    replace("'", '').\
                    replace('.', ' ').\
                    replace(',', '').lower()
    return text

def compute_freq(text, stop_words):
    
    # lower case, split, and filter stop words
    text_list = [word for word in word_tokenize(text, language='english') if word not in stop_words]
    #text_list = text_list.lower()
    #Stemming
    ps = PorterStemmer()
    stemmed_words_text = [ps.stem(w) for w in text_list]
    
    #Frequency distribution in the category
    return FreqDist(stemmed_words_text)

def freq_dict(df, cat, state, stop_words):
    df_sub = df.loc[(df.category_parent_name_ori==cat) & (df.state_grouped==state)]
    text = df_to_text(df_sub)
    return compute_freq(text, stop_words)

def dict_difference(dict1, dict2):
    return {key: value for (key, value) in dict1.items() if key not in dict2}

def freq_unique_words(df, cat, stop_words):
    freq_suc = freq_dict(df, cat, 'successful', stop_words)
    freq_fail = freq_dict(df, cat, 'failed', stop_words)
    freq_suc_u = dict_difference(freq_suc, freq_fail)
    freq_fail_u = dict_difference(freq_fail, freq_suc)
    return freq_suc_u, freq_fail_u

def select_keywords(dictionary, n):
    keywords = [k for (k, v) in sorted(dictionary.items(), key=lambda item: item[1])]
    return keywords[:n]

def get_frequency_score(sentence, freq_suc_u, freq_fail_u, stop_words):
    '''
    #frequence score 
    '''   
    sentence = sentence.replace('?', '').\
                    replace('!', '').\
                    replace('-', ' ').\
                    replace(':', ' ').\
                    replace(';', ' ').\
                    replace('"', '').\
                    replace("'", '').\
                    replace('.', ' ').\
                    replace(',', '').lower()
    
    sentence_words = [word for word in word_tokenize(sentence, language='english') if word not in stop_words]
    # Stemming
    ps = PorterStemmer()
    stemmed_words = [ps.stem(w) for w in sentence_words]
    score_pos = sum([freq_suc_u[word] for word in stemmed_words if word in freq_suc_u])
    score_neg = sum([freq_fail_u[word] for word in stemmed_words if word in freq_fail_u])
    return score_pos - score_neg

# Generate text column
df = pd.read_hdf('../../data/data_english.h5')
df['text'] = df['blurb'] + df['name']
print(df.head())

# Stop_words
stop_words = set(stopwords.words('english'))
stop_words.update([",", "."])


#category
cats = df['category_parent_name_ori'].unique()
print(cats)

#wc = wordcloud.WordCloud(stopwords=stop_words, max_words=2000000)
df_freqs = pd.DataFrame()
for cat in cats:
    print(f'Category: {cat}') 
    freq_suc_u, freq_fail_u = freq_unique_words(df, cat, stop_words)
    #select_keywords(dictionary, n)
    df_sub = df.loc[df.category_parent_name_ori==cat]
    frequency_score = [get_frequency_score(row, freq_suc_u, freq_fail_u, stop_words) for row in df_sub['text']]
    df_sub.loc[:,'frequency_score'] = frequency_score
    df_freqs = pd.concat([df_freqs, df_sub], axis=0)

df_freqs.to_hdf("../../data/data_frequency_score.h5", key="frequency_score")
'''
next = 
id = use as index at the end. Check unique ids/nrow
duplicate data:  df EDA and df timeseries (drop_columns=['category_name', 'category_parent_name',
'location_expanded_country', 'text', 'blurb', 'name','state', 'state_grouped'] in features matrix))
'''