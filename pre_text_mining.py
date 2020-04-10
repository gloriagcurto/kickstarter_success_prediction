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
import numpy as np
from nltk.corpus import stopwords
from nltk import FreqDist
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import os

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
    
    # Tokenize, filter stop words
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
'''
def dict_difference(dict1, dict2):
    return {key: value for (key, value) in dict1.items() if key not in dict2}

def freq_unique_words(df, cat, stop_words):
    freq_suc = freq_dict(df, cat, 'successful', stop_words)
    freq_fail = freq_dict(df, cat, 'failed', stop_words)
    freq_suc_u = dict_difference(freq_suc, freq_fail)
    freq_fail_u = dict_difference(freq_fail, freq_suc)
    return freq_suc_u, freq_fail_u
'''
def select_keywords(dictionary, n):
    keywords = [k for (k, v) in sorted(dictionary.items(), key=lambda item: item[1], reverse = True)]
    return keywords[:n]

def keywords_filter(keywords, dictionary):
    return {key: value for (key, value) in dictionary.items() if key in keywords}

def get_frequency_score(sentence, keywords_suc, keywords_fail, stop_words):
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
    score_pos = sum([keywords_suc[word] for word in keywords_suc if word in stemmed_words])
    score_neg = sum([keywords_fail[word] for word in keywords_fail if word in stemmed_words])
    return (score_pos - score_neg) / len(stemmed_words)

def word_cloud_cat_state(df, cat, state, background_color, stop_words, max_words, colormap):
    df_sub_s = df.loc[df.state_grouped==state]
    text = df_to_text(df_sub_s)
    output_path = os.path.join("../../images/wordclouds/wc_" + cat + "_" + state + ".pdf")
    # Définir le calque du nuage des mots
    wc = WordCloud(background_color=background_color, max_words=max_words, stopwords=stop_words, max_font_size=50, random_state=37, colormap=colormap )
    # Générer et afficher le nuage de mots
    plt.figure() #figsize= (20,10)
    wc.generate(text).to_file(output_path)
    plt.close()

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
    ''' # set difference frequency score
    freq_suc_u, freq_fail_u = freq_unique_words(df, cat, stop_words)
    keywords_suc = keywords_filter(select_keywords(freq_suc_u, 50), freq_suc_u)
    keywords_fail = keywords_filter(select_keywords(freq_fail_u, 50), freq_fail_u)
    df_sub = df.loc[df.category_parent_name_ori==cat]
    word_cloud_cat_state(df_sub, cat, 'successful', 'white', stop_words, 200, 'winter')
    word_cloud_cat_state(df_sub, cat, 'failed', 'white', stop_words, 200, 'autumn_r')
    frequency_score = [get_frequency_score(row, keywords_suc, keywords_fail, stop_words) for row in df_sub['text']]
    df_sub.loc[:,'frequency_score'] = frequency_score
    df_freqs = pd.concat([df_freqs, df_sub], axis=0)
    '''
    freq_suc_u = freq_dict(df, cat, 'successful', stop_words)
    freq_fail_u = freq_dict(df, cat, 'failed', stop_words)
    keywords_suc = keywords_filter(select_keywords(freq_suc_u, 200), freq_suc_u)
    keywords_fail = keywords_filter(select_keywords(freq_fail_u, 200), freq_fail_u)
    df_sub = df.loc[df.category_parent_name_ori==cat]
    word_cloud_cat_state(df_sub, cat, 'successful', 'white', stop_words, 200, 'winter')
    word_cloud_cat_state(df_sub, cat, 'failed', 'white', stop_words, 200, 'autumn_r')
    frequency_score = [get_frequency_score(row, keywords_suc, keywords_fail, stop_words) for row in df_sub['text']]
    df_sub.loc[:,'frequency_score'] = frequency_score
    df_freqs = pd.concat([df_freqs, df_sub], axis=0)

df_freqs.to_hdf("../../data/data_frequency_score.h5", key="frequency_score")
'''
next = 
id = use as index at the end. Check unique ids/nrow
duplicate data:  df EDA and df timeseries (drop_columns=['category_name', 'category_parent_name',
'location_expanded_country', 'text', 'blurb', 'name','state', 'state_grouped'] in features matrix))
'''