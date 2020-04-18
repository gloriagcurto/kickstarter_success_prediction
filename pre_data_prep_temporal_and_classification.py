#!/usr/bin/python
# -*- coding:utf-8 -*-

"""
Pre-processing of Kickstarter_2020-02-13T03_20_04_893Z.
Data preprocesing for time series analysis and classification algorithms
Input Filename:   data_frequency_score.h5
Output Filename:  data_frequency_score.h5

Copyright (c) 2020 Gloria G. Curto.
http://gloriagcurto.info
"""

import pandas as pd
from pandas.api.types import is_numeric_dtype
import numpy as np
import matplotlib.pyplot as plt

def check_numeric (df):
    bo = []
    col_name = []
    results = pd.DataFrame()
    for col in df.columns:
        bo.append(is_numeric_dtype(df[col]))
        col_name.append(col)
    
    results = results.append([col_name, bo])
    results_transpose = results.transpose()
    results_transpose.columns = ['col_name', 'bo']
    return(results_transpose)


df = pd.read_hdf('../../data/data_frequency_score.h5')
cols = [col for col in df.columns]
print(cols)
print(f'Duplicates: {df.duplicated().sum()}')
df.drop_duplicates(keep='first', inplace=True)

columns_to_drop = ['blurb', 'created_at', 'deadline', 'id', 'launched_at', 'name', 'state', 'usd_pledged',
                   'currency_orig','category_name_ori', 'category_parent_name_ori', 'location_expanded_country_ori', 
                   'text']

print(f'Dimensions of data_frequency_score.h5: {df.shape}')
print(f'Columns to drop: {len(columns_to_drop)}')

df_time_series = df.drop(columns_to_drop, axis = 1)
print(f'Dimensions of df_time_series: {df_time_series.shape}')

print(f'{df.shape[1]-len(columns_to_drop)} = {df_time_series.shape[1]}')

# Group by date, count successful, failed, and total projects by day (state_changed_at)

df_state_time_course = df_time_series.loc[:,['state_changed_at', 'state_grouped']]
print(f'Dimensions: {df_state_time_course.shape}')

#Total projects
df_counts = df_state_time_course['state_changed_at'].value_counts().sort_index().reset_index()
df_counts.columns = ['state_changed_at','project_count']
df_counts.head

#Successful projects
df_success = df_state_time_course.loc[df_state_time_course['state_grouped'] == 'successful']
print(f'Dimensions success : {df_success.shape}')
df_success_counts = df_success['state_changed_at'].value_counts().sort_index().reset_index()
df_success_counts.columns = ['state_changed_at','project_count']

#Failed projects
df_failed = df_state_time_course.loc[df_state_time_course['state_grouped'] == 'failed']
print(f'Dimensions failed : {df_failed.shape}')
df_failed_counts = df_failed['state_changed_at'].value_counts().sort_index().reset_index()
df_failed_counts.columns = ['state_changed_at','project_count']

#Plot total number; successful and failed project counts by date
# share x and y axis
ax1 = plt.subplot(311)
plt.plot('state_changed_at', 'project_count', data=df_counts, color = '#343434', label = 'All')
plt.title('Temporal distribution of projects')
plt.ylabel('Counts')
plt.legend()
plt.setp(ax1.get_xticklabels(), fontsize=6)

# share x and y
ax2 = plt.subplot(312, sharex=ax1, sharey=ax1)
plt.plot('state_changed_at', 'project_count', data=df_success_counts, color='#66cdaa', label = 'Successful')
plt.ylabel('Counts')
plt.legend()
plt.setp(ax1.get_xticklabels(), visible=False)
# make these tick labels invisible
plt.setp(ax2.get_xticklabels(), visible=False)

# share x and y
ax3 = plt.subplot(313, sharex=ax1, sharey=ax1)
plt.plot('state_changed_at', 'project_count', data=df_failed_counts, color='#ff8b61', label='Failed')
plt.ylabel('Counts')
plt.legend()
plt.tight_layout()
plt.savefig("../../images/nprojects_date_state_change.pdf")

print('I do not see any time structure just by eye. Data are not evenly spaced and I cannot perform seasonal decomposition.')

# Remove additional columns to obtain a numeric df (including target) for classification
print(f'Dimensions before column drop: {df_time_series.shape}')
df_time_series.drop(['state_changed_at', 'state_grouped'], axis=1, inplace=True)
print(f'Dimensions after column drop: {df_time_series.shape}')

#Check if we have a numeric df
results_numeric = check_numeric(df_time_series)
print(results_numeric.columns)
print(f'Non numeric columns: {results_numeric.loc[results_numeric.bo ==False]}')

#df_time_series.to_hdf("../../data/data_classification.h5", key="classification")
'''
'''