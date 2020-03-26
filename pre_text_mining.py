#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Pre-processing of Kickstarter_2020-02-13T03_20_04_893Z.
Text mining
Input Filename: data_wo_text_mining.h5
Output Filename: data_text_mining.h5

Copyright (c) 2020 Gloria G. Curto.
http://gloriagcurto.info
'''

import pandas as pd
import googletrans
from googletrans import Translator
from langdetect import detect

data_wo_text_mining = pd.read_hdf('../../data/data_wo_text_mining.h5')

# Explore languages
print(data_wo_text_mining['location_expanded_country_ori'].unique())

# Explore language in variables for text mining
blurbs = data_wo_text_mining.groupby('location_expanded_country_ori')['blurb'].first()
blurbs = pd.concat([blurbs, data_wo_text_mining.groupby('location_expanded_country_ori')['blurb'].last()], axis = 0)
blurbs.to_csv("../../data/blurbs_by_country. csv", sep ="\t")

names = data_wo_text_mining.groupby('location_expanded_country_ori')['name'].first()
names = pd.concat([names, data_wo_text_mining.groupby('location_expanded_country_ori')['name'].last()], axis = 0)
names.to_csv("../../data/names_by_country. csv", sep ="\t")


# Translate blurb and name to english

'''
Keep language detected, original text and translated
From language detected compute a variable 'English' with english = 1, other languages = 0
'''
'''
def translate_to_english (df):
    '''
    #Translate to english a df containing only the columns to translate
    '''
    translated = pd.DataFrame()
    translator = Translator()
    trans_df = pd.DataFrame(columns=['en_text', 'language'])
    
    for cname in df.columns: 
        print(cname)
        for row in df[cname]:
            # Routing of english versus non english  
            lang = detect(row)
            if lang == 'en':
                trans_df.append([[row], ['en']], ignore_index=True)
            else:
                # REINITIALIZE THE API due to a JSON error
                translator = Translator()
                try:
                    # translate the 'text' column
                    trans = translator.translate(row, dest='en')
                    trans_df = pd.concat([trans.text, trans.extra_data['original-language']], axis=0)
                                     
                except Exception as e:
                    print(str(e), row)
                    continue
                trans_df['new_index'] = range(len(trans))
                trans_df.set_index('new_index', inplace=True)
                trans_df.columns =  map(lambda x: df[cname].name + '_translated' + x , trans_df.columns)
                translated =  pd.concat([translated, trans_df], axis=1)

    return translated

data_trans = data_wo_text_mining.loc[:,['blurb', 'name']]
print(f'Dimensions data_trans: {data_trans.shape}')

translated = translate_to_english(data_trans)

'''

#data = ['Dobrý deň', 'majestátny orol', 'krehká dohoda']

#translated = translator.translate(data_wo_text_mining['blurb'][3], dest='en')

#for trans in translated:
#    print(f'{trans.origin} -> {trans.text}')


#Detect language
text1 = 'A Római Birodalom (latinul Imperium Romanum) az ókori Róma által létrehozott államalakulat volt a Földközi-tenger medencéjében'


text2 = 'Vysoké Tatry sú najvyššie pohorie na Slovensku a v Poľsku a sú zároveň jediným horstvom v týchto štátoch s alpským charakterom.'

translator = Translator()

dt1 = translator.detect(text1)
print(dt1)

dt2 = translator.detect(text2)
print(dt2)
'''
