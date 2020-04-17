#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Pre-processing of Kickstarter_2020-02-13T03_20_04_893Z.
plots EDA feature importance: frequency_score, spotlight, profile, usd_goal
Input Filename: train, test
Output Filename: plots

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

'''
def counts_dfs (df, variable_name):
    #Total projects
    df_counts = df_state_sec_cat['category_name_ori'].value_counts().sort_index().reset_index()
    df_counts.columns = ['cat','project_count']

    #Successful projects
    df_success = df_state_sec_cat.loc[df_state_sec_cat['state_grouped'] == 'successful']
    print(f'Dimensions success : {df_success.shape}')
    df_success_counts = df_success['category_name_ori'].value_counts().sort_index().reset_index()
    df_success_counts.columns = ['cat','project_count']

    #Failed projects
    df_failed = df_state_sec_cat.loc[df_state_sec_cat['state_grouped'] == 'failed']
    print(f'Dimensions failed : {df_failed.shape}')
    df_failed_counts = df_failed['category_name_ori'].value_counts().sort_index().reset_index()
    df_failed_counts.columns = ['cat','project_count']

return df_counts, df_success_counts, df_failed_counts
'''


train = pd.read_hdf("../../data/train_frequency_score.h5")
test = pd.read_hdf("../../data/test_frequency_score.h5")
df = pd.concat([test, train], axis=0)
print(df.head(3))
df_state_sec_cat = df.loc[:,['category_name_ori', 'state_grouped']]
print(f'Dimensions: {df_state_sec_cat.shape}')
print('Number of secondary category levels : {df.category_name_ori.value_counts()}')

#function counts df

# plot counts by categories and state
output_path = "../../images/variable_pruning_EDA/cat_name/nprojects_" 
var_to_plot = 'category_name_ori'
title = 'Parent category:'
xlab='Secondary category'
ylab='Project count'
palette_dict = dict(successful='#66cdaa', failed='#ff8b61')

for cat in df.category_parent_name_ori.unique():
    df_sub = df[df['category_parent_name_ori']==cat]
    print(cat)
    plot_count_by_state(df_sub, var_to_plot, cat, palette_dict, title, xlab, ylab, output_path)



