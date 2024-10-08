#!/usr/bin/python
# -*- coding:utf-8 -*-

"""
Pre-processing of Kickstarter_2020-02-13T03_20_04_893Z.
Text mining. Filter projects with blurb or name not in English
Input Filename: data_wo_text_mining_currency.h5, words_dictionary.json (from https://github.com/dwyl/english-words/blob/master/words_dictionary.json)
Output Filename: data_english.h5

Copyright (c) 2020 Gloria G. Curto.
http://gloriagcurto.info
"""

import json
import sys
import pandas as pd
from nltk.tokenize import word_tokenize

def get_english_score(sentence):
    '''
    Score to classify a sentence as English
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
    
    sentence_words = [word for word in word_tokenize(sentence, language='english')]
    valid_words = [word for word in sentence_words if len(word) > 3]
    if not valid_words:
        return 0
    return sum([(word in all_words) for word in valid_words]) / len(valid_words)


with open("../../data/words_dictionary.json", 'rt') as f:
    json_words = json.load(f)

all_words = json_words.keys()
print(len(all_words))
'''
#test
print(get_english_score('hello what is your name dude?'))
print(get_english_score('This is a second project on building a new platform'))
print(get_english_score('Este es el título de mi proyecto de análisis de datos'))
print(get_english_score("C'est le titre de mon projet d'analyse de données"))
print(get_english_score("Dies ist der Titel meines Datenanalyseprojekts"))
print(get_english_score("Dies ist ein zweites Projekt zum Aufbau einer neuen Plattform"))
print(get_english_score("Tämä on toinen projekti uuden alustan rakentamiseksi"))
'''
# Fix a threshold at 0.8

'''
Filter rows with an score lower than 0.8 in blurb or name
'''

df = pd.read_hdf('../../data/data_wo_text_mining_currency.h5')
print(df.head(3))
english_df = pd.DataFrame()
keep = []    
for row in df.itertuples(index=False):
    blurb_score = get_english_score(row[1]) # 'blurb'.index = 1
    name_score = get_english_score(row[7]) # 'name'.index = 7
    # Write row
    if blurb_score >= 0.8 and name_score >=0.8 :
        keep.append(row)

english_df = pd.DataFrame(keep, columns=df.columns)

print(f'Dimensions of English filtered data: {english_df.shape}')
print(f'Dimensions of non filtered data: {df.shape}')
english_df.to_hdf("../../data/data_english.h5", key="english")

'''
Next:
pre_split_train_test, then text mining
'''