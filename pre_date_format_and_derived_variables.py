#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Pre-processing of Kickstarter_2020-02-13T03_20_04_893Z.
Dates and derived variables pre-procesing in columns not JSON encoded.

Copyright (c) 2020 Gloria G. Curto.
http://gloriagcurto.info
'''

import pandas as pd
import numpy as np

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
#print(f'Tail: {kcks.tail()}')

# Columns in JSON format are: category, creator, location, photo, profile, urls. They are treated separately for now.

# Drop the JSON format columns into a new df
no_json_df = kcks.drop(['category', 'creator', 'location', 'photo', 'profile', 'urls'], axis=1)
print(no_json_df.head())

# Drop country and country_displayable_name because they are redundant with varaibles extracted from the JSON encoded columns.
# Drop currency_symbol
no_json_df.drop(['country', 'country_displayable_name', 'currency_symbol'], axis=1, inplace=True)
print(no_json_df.iloc[:,0:10].head())

# Drop source_url	
no_json_df.drop(['source_url'], axis=1, inplace=True)
no_json_df.iloc[:,10:20].head()
no_json_df.iloc[:,20:30].head()

# Convert unix dates to datetime in 'created_at', 'deadline', 'launched_at', 'state_changed_at'
print(no_json_df.columns)

date_cols = ['created_at', 'deadline', 'launched_at', 'state_changed_at']

for column in date_cols:
        no_json_df[column] = pd.to_datetime(no_json_df[column],yearfirst=True, unit='s').dt.normalize() #normalize removes time

print(no_json_df.iloc[:,0:15].head())

print(no_json_df.iloc[:,15:25].head())

# Weekday columns for each date variable

date_cols = ['created_at', 'deadline', 'launched_at', 'state_changed_at']

for column in date_cols:
    new = "weekday_" + no_json_df[column].name 
    no_json_df[new] = no_json_df[column].dt.weekday
print(no_json_df.columns)
print(no_json_df.iloc[0:4,20:30])

#  Month columns for each date variable

date_cols = ['created_at', 'deadline', 'launched_at', 'state_changed_at']

for column in date_cols:
    new = "month_" + no_json_df[column].name 
    no_json_df[new] = no_json_df[column].dt.month
print(no_json_df.columns)
print(no_json_df.iloc[0:4,27:40])

# Year columns for each date variable

date_cols = ['created_at', 'deadline', 'launched_at', 'state_changed_at']

for column in date_cols:
    new = "year_" + no_json_df[column].name 
    no_json_df[new] = no_json_df[column].dt.year
print(no_json_df.columns)
print(no_json_df.iloc[0:4,30:45])

# Compute initial_found_rising_duration

no_json_df["initial_found_rising_duration"] = (no_json_df['deadline']- no_json_df["launched_at"])/np.timedelta64(1,'D')

print(no_json_df.iloc[0:4,35:40])

#Compute found_rising_duration

no_json_df["found_rising_duration"] = (no_json_df['state_changed_at']- no_json_df["launched_at"])/np.timedelta64(1,'D')

print(no_json_df.iloc[0:4,35:40])

print(no_json_df.columns)

no_json_df.to_hdf("../../data/no_json_df_dates_variables.h5", key = 'dates_variables')

'''
Next:
Currency and other variables
'''