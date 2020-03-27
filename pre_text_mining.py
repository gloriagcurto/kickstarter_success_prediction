#!/usr/bin/python
# -*- coding:utf-8 -*-

"""
Pre-processing of Kickstarter_2020-02-13T03_20_04_893Z.
Text mining. Tokenize, frecuent words by category
Input Filename:  data_english.h5
Output Filename: data_text_mining.h5

Copyright (c) 2020 Gloria G. Curto.
http://gloriagcurto.info
"""

import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

def tokenize_columns (df, cnames):
    '''
    tokenize words in a set of columns
    '''
    word_tok = pd.DataFrame()
    text_min_df = pd.DataFrame()
    for cname in cnames: 
        print(cname)
        row_words = [word_tokenize(row, language='english') for row in df[cname] ]
        word_tok = pd.concat([word_tok, row_words], axis=0)
        word_tok['new_index'] = range(len(row_words))
        word_tok.set_index('new_index', inplace=True)
        word_tok.columns =  map(lambda x: df[cname].name + '_word_tok' + x , word_tok.columns)
        text_min_df =  pd.concat([text_min_df, word_tok], axis=1)              
    return text_min_df



df = pd.read_hdf('../../data/data_english.h5')
text_min_df = tokenize_columns(df, ['blurb', 'name'])

# Stop words filtering
# Initialize stop words
stop_words = set(stopwords.words('english'))
print(stop_words)
stop_words.update([",", "."])

# stop_words_filtering
def stop_words_filtering(df) : 
    
    for cname in df.columns:
        new_cname =  map(lambda x: df[cname].name + '_wo_stop' + x)
        df[new_cname] = df[cname].apply(lambda x: [word for word in x if word not in stop_words])
        print(new_cname)
    return df

text_min_wo_stop_df = stop_words_filtering(text_min_df)
    
#lemmanisation

#Split by category

#Frequency by category and failed/successful

#index frequent words failed/success

#wordcloud by category and failed successful

#put indexes back with the rest of variables

#text_min_df.to_hdf("../../data/text_mining_subset.h5", key="english")

'''
Next: user_rate (based on project failure(-1)/success(+1) user history)
'''