#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Pre-processing of Kickstarter_2020-02-13T03_20_04_893Z.
Decode joined data.

Copyright (c) 2020 Gloria G. Curto.
http://gloriagcurto.info
'''

import pandas as pd
import json

def deserialize_in_batch (df):
    '''
    Deserialize JSON columns in batch
    '''
    deserialized = pd.DataFrame()
    for cname in df.columns: 
        print(cname)
        frames = [ pd.json_normalize(json.loads(row)) for row in df[cname] ]
        cname_df = pd.concat(frames, axis=0)
        cname_df['new_index'] = range(len(frames))
        cname_df.set_index('new_index', inplace=True)
        cname_df.columns =  map(lambda x: df[cname].name + '_' + x , cname_df.columns)
        deserialized =  pd.concat([deserialized, cname_df], axis=1)              
    return deserialized

#h = pd.read_hdf("../../data/deserialized.h5", key="kickstarter")
kcks = pd.read_csv("../../data/joined.csv")
print(f'Original shape: {kcks.shape}')
print(f'Original columns: {kcks.columns}')

# Remove columns and rows with missing information
kcks.drop(["friends", "is_backing", "is_starred", "permissions"], axis=1, inplace=True)
print(f'Shape after drop empty columns: {kcks.shape}') 
kcks.dropna( axis=0, how='any', inplace=True)
kcks.reset_index(drop=True, inplace=True)
print(f'Shape after drop rows with na: {kcks.shape}')

print(f'Shape after drop rows with na: {kcks.shape}')
print(f'Tail: {kcks.tail()}')

# Some columns are JSON encoded. Deserialize them in order to retrieve information.
# Columns in JSON format are: category, creator, location, photo, profile, urls.

# Subset the JSON format columns into a new df
json_df = kcks.loc[:,['category', 'creator', 'location', 'photo', 'profile', 'urls']]
print(json_df.head())


deserialized_df = deserialize_in_batch(json_df)
#deserialized_df = kcks[['backers_count','blurb', 'country']]

deserialized_df.to_hdf('../../data/deserialized.h5', key='kickstarter', mode='w')

print(f'Shape of deserialized: {deserialized_df.shape}')
print(f'Columns of deserialized: {deserialized_df.columns}')

