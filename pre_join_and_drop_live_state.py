#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Pre-processing of Kickstarter_2020-02-13T03_20_04_893Z.
Join JSON decoded to the rest of data.
Drop 'live' status.
Input Filename: deserialized_dummied.h5, no_json_df_dates_other_variables.h5
Output Filename: data_wo_text_mining.h5

Copyright (c) 2020 Gloria G. Curto.
http://gloriagcurto.info
'''

import pandas as pd

json_df = pd.read_hdf('../../data/deserialized_dummied.h5')

print(json_df.columns, json_df.shape)
print(json_df.head())

no_json_df = pd.read_hdf('../../data/no_json_df_dates_other_variables.h5')
print(no_json_df.columns, no_json_df.shape)
print(no_json_df.head())

data= pd.concat([no_json_df, json_df], axis=1)
print(data.shape)
print(data.head())

# Drop 'live' state rows

print(f'Dimensions before live state dropping: {data.shape}')

data.drop(data[data.state_grouped == 'live'].index, inplace=True)

print(f'Dimensions after live state dropping: {data.shape}')

# Output file

data.to_hdf("../../data/data_wo_text_mining.h5", key = 'data_wo_text_mining')

'''
next:
check and remove duplicates
# 
# id = use as index at the end
# 
# Remember to drop columns=['category_name_ori', 'category_parent_name_ori',
# 'location_expanded_country_ori', 'currency_orig'] in features matrix
'''
