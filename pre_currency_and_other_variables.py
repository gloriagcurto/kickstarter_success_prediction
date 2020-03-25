#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Pre-processing of Kickstarter_2020-02-13T03_20_04_893Z.
Currency, country, binary encoding, one hot encoding of no JSON encoded columns, state pre-procesing.

Copyright (c) 2020 Gloria G. Curto.
http://gloriagcurto.info
'''

import pandas as pd
import numpy as np

no_json_dates= pd.read_hdf("../../data/no_json_df_dates_variables.h5", key = 'dates_variables')

print(no_json_dates.shape)
print(no_json_dates.columns)
no_json_dates.head()

# Exploration
print(no_json_dates.iloc[:,:25].agg(['count', 'size', 'nunique']))

print(no_json_dates.iloc[:,:25].isnull().sum())

no_json_dates.head()

#drop: 'current_currency', 'currency_trailing_code', 'fx_rate', 'converted_pledged_amount', 'pledged', 'slug', 'usd_type'
no_json_dates.drop(['current_currency', 'currency_trailing_code', 'fx_rate', 'converted_pledged_amount', 'pledged', 'slug', 'usd_type'], axis=1, inplace=True)
print(no_json_dates.columns)


# Convert goal to usd using static_usd_rate
no_json_dates['usd_goal'] = no_json_dates['goal']*no_json_dates['static_usd_rate']
no_json_dates['usd_goal'].head()

#No need: goal and static_usd_rate
no_json_dates.drop(columns=['goal', 'static_usd_rate'], inplace=True)

# Drop 'is_starrable' because I don't know what it is
no_json_dates.drop(columns=['is_starrable'], inplace=True)

#binary encoding : 'disable_communication', 'spotlight', 'staff_pick'
for column in ['disable_communication', 'spotlight', 'staff_pick']:
    no_json_dates[column].replace([False,True],[0, 1], inplace=True)

no_json_dates.iloc[:,5:15].head()

# Dummification currency
no_json_dates_dummied = pd.get_dummies(no_json_dates, prefix_sep='_', columns=['currency'], drop_first=True)

print(f'New column names: {no_json_dates_dummied.columns}')
print(f'New dimmensions: {no_json_dates_dummied.shape}')
no_json_dates_dummied.head()

# 'state' pre-processing
print(no_json_dates_dummied.state.agg(['count', 'size', 'nunique']))

print(no_json_dates_dummied.state.unique())

# Group ['failed' 'canceled' 'suspended'] as 'failed' in 'state_grouped'
no_json_dates_dummied['state_grouped'] = no_json_dates_dummied['state'].replace(['canceled', 'suspended'],['failed', 'failed'])

# Code failed = 0, successful=1, live=2. Decide what to do with 'live' after visualization (probably to drop)
no_json_dates_dummied['state_code'] = no_json_dates_dummied['state_grouped'].replace(['failed','successful', 'live'],[0, 1, 2])
no_json_dates_dummied.iloc[:,25:].head()

# Visualization of state subcategories
print(no_json_dates_dummied['state_grouped'].agg(['count', 'size', 'nunique']))

print(no_json_dates_dummied['state_grouped'].value_counts())


no_json_dates_dummied.to_hdf("../../data/no_json_df_dates_other_variables.h5", key = 'other_variables')


'''
Next:
There are 6651 projects with 'live' status. Because they are not closed, they are not interesting for the model
and those rows have to be drop when we join all data together.
blurb, name = text mining
id = use as index at the end
'''