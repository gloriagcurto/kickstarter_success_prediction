#!/usr/bin/python
# -*- coding:utf-8 -*-

"""
Pre-processing of Kickstarter_2020-02-13T03_20_04_893Z.
Text mining. Tokenize, frequent words by category
Input Filename:  data_english.h5
Output Filename: data_text_mining.h5

Copyright (c) 2020 Gloria G. Curto.
http://gloriagcurto.info
"""
import pandas as pd
import wordcloud
import numpy as np
from nltk.corpus import stopwords

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

    return np.mean([freqs[word] for word in sentence_words])


# Failed/successful:
f_s = ['successful', 'failed']

print(df.columns[40:40])
#category
cats = df['category_parent_name_ori'].unique()
print(cats)



wc = wordcloud.WordCloud(stopwords=stop_words)

for cat in cats:
    for s in f_s:
        df_sub = df[(df['category_parent_name_ori']==cat) & (df['state']==s)]
        text = ""
        for comment in df_sub.text : 
            text += comment
        # lower case
        wc.generate(text.lower())
        for row in df_sub['text']:
            df_sub['frequency_score'] = get_frequency_score(row, wc.words_, stop_words)

'''
Next: user_rate (based on project failure(-1)/success(+1) user history)
'''