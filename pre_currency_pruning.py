
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Pre-processing of Kickstarter_2020-02-13T03_20_04_893Z.
currency variables pruning
Input Filename: data_wo_text_mining_country.h5
Output Filename: data_wo_text_mining_currency.h5

Copyright (c) 2020 Gloria G. Curto.
http://gloriagcurto.info
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_count_by_state(df, var_to_plot, cat, palette_dict, title, xlab, ylab, output_path):
    output_filename = output_path + cat + "_state.pdf"
    fig = plt.figure(figsize=(20,5))
    g = sns.countplot(x=var_to_plot, hue='state_grouped', data=df_sub, palette=palette_dict)
    g.set_title(f' {title} {cat}')
    g.set_xlabel(xlab)
    g.set_ylabel(ylab)
    plt.legend(title='Project state', loc='upper left')
    plt.savefig(output_filename)
    plt.close(fig)


df = pd.read_hdf('../../data/data_wo_text_mining_country.h5')

print(df.head(3))
df_state_currency = df.loc[:,['currency_orig', 'state_grouped']]
print(f'Dimensions: {df_state_currency.shape}')

#Total projects
df_counts = df_state_currency['currency_orig'].value_counts().sort_index().reset_index()
df_counts.columns = ['currency','project_count']

#Successful projects
df_success = df_state_currency.loc[df_state_currency['state_grouped'] == 'successful']
print(f'Dimensions success : {df_success.shape}')
df_success_counts = df_success['currency_orig'].value_counts().sort_index().reset_index()
df_success_counts.columns = ['currency','project_count']

#Failed projects
df_failed = df_state_currency.loc[df_state_currency['state_grouped'] == 'failed']
print(f'Dimensions failed : {df_failed.shape}')
df_failed_counts = df_failed['currency_orig'].value_counts().sort_index().reset_index()
df_failed_counts.columns = ['currency','project_count']


df_counts_200 = df_counts.loc[df_counts.project_count >= 200]
print(f'Countries with more than 200 projects : {df_counts_200.shape[0]}')
print(f'Countries: {df_counts.shape[0]}')

df_success_counts_200 = df_success_counts.loc[df_success_counts.project_count >= 200]
print(f'Countries with more than 200 success projects : {df_success_counts_200.shape[0]}')
print(f'Countries: {df_counts.shape[0]}')

df_failed_counts_200 = df_failed_counts.loc[df_failed_counts.project_count >= 200]
print(f'Countries with more than 200 failed projects : {df_failed_counts_200.shape[0]}')
print(f'Countries: {df_counts.shape[0]}')

#plot and compute percentages of successful projects by secondary category to decide one hot encoded columns under category_name to drop.

# plot counts by categories and state
output_path = "../../images/variable_pruning_EDA/currency/nprojects_" 
var_to_plot = 'currency_orig'
title = 'Currency'
xlab=''
ylab='Project count'
palette_dict = dict(successful='#66cdaa', failed='#ff8b61')

for currency in df.currency_orig.unique():
    df_sub = df[df['currency_orig']==currency]
    #print(currency)
    plot_count_by_state(df_sub, var_to_plot, currency, palette_dict, title, xlab, ylab, output_path)

# Percentage success

percen = []
currencies = []
for currency in df.currency_orig.unique() : 
    df_sub = df[df['currency_orig']==currency]
    currencies.append(currency)
    percen.append((df_sub[df_sub['state_grouped']=="successful"].size/df_sub.size)*100)

p_success = pd.DataFrame({'currencies': currencies, 'percentages':percen})
p_success.to_csv('../../var_pruning_notebooks/p_success_currency.csv')

print(p_success.head(10))

print(f'Percentage of success: {(df[df.state_grouped=="successful"].size/df.size)*100}')
print(f'Percentage of failure: {(df[df.state_grouped=="failed"].size/df.size)*100}')

# Keep currencies with a percentage of successful projects of at least 60% 
keep_p = p_success[p_success.percentages >= 60]
print(f'Number of currencies with more than 60% of successful projects: {len(keep_p)}')
keep = keep_p.merge(right=df_counts, how='inner', left_on='currencies', right_on='currency')
keep.to_csv('../../data/keep_currencies_60success.csv')
'''
Keep currencies with more than 60% of successful projects ands more than 50 projects:
currencies:
GBP, HKD, SGD, JPY 

These are currencies strongly associated with countries that we have already included as indicative variables. Drop all currency related columns

'''
cols = [col for col in df.columns]
#print(cols)

with open('../../data/colnames.txt', "w") as output:
    output.write(str(cols))


to_drop = ['currency_CAD', 'currency_CHF', 'currency_DKK', 'currency_EUR', 'currency_GBP', 'currency_HKD', 'currency_JPY', 'currency_MXN', 'currency_NOK', 'currency_NZD', 'currency_SEK', 'currency_SGD', 'currency_USD']

df.drop(to_drop, axis=1, inplace=True)

print(f'Dimensions after pruning: {df.shape}')

df.to_hdf("../../data/data_wo_text_mining_currency.h5", key = 'country_pruning')

'''
size: 119
Next: filter english texts with pre_words.py
'''

