#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Pre-processing of Kickstarter_2020-02-13T03_20_04_893Z.
category_name variables pruning (secondary categories)
Input Filename: data_wo_text_mining_profile.h5
Output Filename: data_wo_text_mining_cat.h5

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

df = pd.read_hdf('../../data/data_wo_text_mining_profile.h5')

print(df.head())

df_state_sec_cat = df.loc[:,['category_name_ori', 'state_grouped']]
print(f'Dimensions: {df_state_sec_cat.shape}')
print(f'Number of secondary category levels : {df.category_name_ori.value_counts()}')

#Total projects
df_counts = df_state_sec_cat['category_name_ori'].value_counts().sort_index().reset_index()
df_counts.columns = ['cat','project_count']
df_counts.head
#df_counts.plot()
#Successful projects
df_success = df_state_sec_cat.loc[df_state_sec_cat['state_grouped'] == 'successful']
print(f'Dimensions success : {df_success.shape}')
df_success_counts = df_success['category_name_ori'].value_counts().sort_index().reset_index()
df_success_counts.columns = ['cat','project_count']
#df_success_counts.plot()
#Failed projects
df_failed = df_state_sec_cat.loc[df_state_sec_cat['state_grouped'] == 'failed']
print(f'Dimensions failed : {df_failed.shape}')
df_failed_counts = df_failed['category_name_ori'].value_counts().sort_index().reset_index()
df_failed_counts.columns = ['cat','project_count']
#df_failed_counts.plot()

# More than 1000 projects
df_counts_1000 = df_counts.loc[df_counts.project_count >= 1000]
print(f'Category_name with more than 1000 projects : {df_counts_1000.shape[0]}')
print(f'Category_name: {df_counts.shape[0]}')

# More than 500 projects

df_counts_500 = df_counts.loc[df_counts.project_count >= 500]
print(f'Category_name with more than 500 projects : {df_counts_500.shape[0]}')
print(f'Category_name: {df_counts.shape[0]}')

df_success_counts_500 = df_success_counts.loc[df_success_counts.project_count >= 500]
print(f'Category_name with more than 500 success projects : {df_success_counts_500.shape[0]}')
print(f'Category_name: {df_counts.shape[0]}')

df_failed_counts_500 = df_failed_counts.loc[df_failed_counts.project_count >= 500]
print(f'Category_name with more than 200 failed projects : {df_failed_counts_500.shape[0]}')
print(f'Category_name: {df_counts.shape[0]}')

'''
# plot total counts, successful and failed projects
# total counts
plt.figure(figsize=(30,13))
ax1 = plt.subplot(311)
plt.scatter(df_counts_500.cat, df_counts_500.project_count, c='k')
plt.setp(ax1.get_xticklabels(), fontsize=12)

# successful projects
ax2 = plt.subplot(312)
plt.scatter(df_success_counts_500.cat, df_success_counts_500.project_count, c='#66cdaa')
# make these tick labels invisible
plt.setp(ax2.get_xticklabels(), fontsize=12)

# failed projects
ax3 = plt.subplot(313)
plt.scatter(df_failed_counts_500.cat, df_failed_counts_500.project_count, c='#ff8b61')
plt.setp(ax3.get_xticklabels(), fontsize=12)
plt.savefig("../../images/variable_pruning_EDA/cat_name/nprojects_scatter_cat_name_state_500.svg", format="svg")
plt.show()
'''
#plot and compute percentages of successful projects by secondary category to decide one hot encoded columns under category_name to drop.

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


print(df.groupby(['category_parent_name_ori', 'category_name_ori' ]).size())
df.groupby(['category_parent_name_ori', 'category_name_ori' ]).size().to_csv('../../images/variable_pruning_EDA/cat_name/cats_counts.csv')
df.groupby(['category_parent_name_ori', 'category_name_ori', 'state_grouped', ]).size().to_csv('../../images/variable_pruning_EDA/cat_name/cats_counts_state.csv')


# Percentage success

percen = []
cats = []
cats_parent = []

for cat_p in df.category_parent_name_ori.unique() :
    df_s = df[df['category_parent_name_ori']==cat_p]
    for cat in df_s.category_name_ori.unique() : 
        df_sub = df_s[df_s['category_name_ori']==cat]
        cats.append(cat)
        cats_parent.append(cat_p)
        percen.append((df_sub[df_sub['state_grouped']=="successful"].size/df_sub.size)*100)


p_success = pd.DataFrame({'categories_parent': cats_parent, 'categories': cats, 'percentages':percen})
p_success.to_csv('../../var_pruning_notebooks/p_success_categories.csv')

print(p_success.head(10))

print(f'Percentage of success: {(df[df.state_grouped=="successful"].size/df.size)*100}')
print(f'Percentage of failure: {(df[df.state_grouped=="failed"].size/df.size)*100}')

# Keep categories with a percentage of successful projects of at least 55% and informative within parent categories 
keep = p_success[p_success.percentages >= 55]
print(len(keep))

kp = [k for k in keep.categories]
print(kp) 
print(keep.categories.unique().size)
# After verification of plots and percentages, all comic, theater, dance secondary categories are successful, it is not informative to keep the secondaries.

cols = [col for col in df.columns]
#print(cols)

with open('../../data/colnames.txt', "w") as output:
    output.write(str(cols))


print(f'Dimensions before pruning: {df.shape}')

to_drop = ['category_name_Academic', 'category_name_Action', 'category_name_Animals', 'category_name_Animation', 'category_name_Apps', 'category_name_Architecture', 'category_name_Art', 'category_name_Audio', 'category_name_Bacon', 'category_name_Blues', 'category_name_Calendars', 'category_name_Camera Equipment', 'category_name_Candles', 'category_name_Ceramics', 'category_name_Childrenswear', 'category_name_Civic Design', 'category_name_Comic Books', 'category_name_Comics', 'category_name_Community Gardens', 'category_name_Conceptual Art', 'category_name_Cookbooks', 'category_name_Couture', 'category_name_Crochet', 'category_name_DIY', 'category_name_DIY Electronics', 'category_name_Dance', 'category_name_Digital Art', 'category_name_Drama', 'category_name_Drinks', 'category_name_Electronic Music', 'category_name_Embroidery', 'category_name_Events', 'category_name_Experimental', 'category_name_Fabrication Tools', 'category_name_Faith', 'category_name_Family', 'category_name_Fantasy', "category_name_Farmer's Markets", 'category_name_Farms', 'category_name_Festivals', 'category_name_Fine Art', 'category_name_Flight', 'category_name_Food Trucks', 'category_name_Footwear', 'category_name_Gaming Hardware', 'category_name_Glass', 'category_name_Graphic Design', 'category_name_Graphic Novels', 'category_name_Hip-Hop', 'category_name_Horror', 'category_name_Immersive', 'category_name_Installations', 'category_name_Interactive Design', 'category_name_Jewelry', 'category_name_Kids', 'category_name_Knitting', 'category_name_Latin', 'category_name_Literary Journals', 'category_name_Live Games', 'category_name_Makerspaces', 'category_name_Metal', 'category_name_Mixed Media', 'category_name_Mobile Games', 'category_name_Movie Theaters', 'category_name_Music Videos', 'category_name_Musical', 'category_name_Nature', 'category_name_People', 'category_name_Performances', 'category_name_Periodicals', 'category_name_Pet Fashion', 'category_name_Photo', 'category_name_Photobooks', 'category_name_Places', 'category_name_Plays', 'category_name_Poetry', 'category_name_Pottery', 'category_name_Print', 'category_name_Printing', 'category_name_Punk', 'category_name_Quilts', 'category_name_R&B', 'category_name_Radio & Podcasts', 'category_name_Ready-to-wear', 'category_name_Residencies', 'category_name_Restaurants', 'category_name_Robots', 'category_name_Romance', 'category_name_Science Fiction', 'category_name_Sculpture', 'category_name_Small Batch', 'category_name_Software', 'category_name_Sound', 'category_name_Space Exploration', 'category_name_Spaces', 'category_name_Stationery', 'category_name_Taxidermy', 'category_name_Television', 'category_name_Textiles', 'category_name_Theater', 'category_name_Thrillers', 'category_name_Translations', 'category_name_Vegan', 'category_name_Video', 'category_name_Video Art', 'category_name_Wearables', 'category_name_Weaving', 'category_name_Web', 'category_name_Webcomics', 'category_name_Woodworking', 'category_name_Workshops', 'category_name_World Music', 'category_name_Young Adult', 'category_name_Zines']
df.drop(to_drop, axis=1, inplace=True)

print(f'Dimensions after pruning: {df.shape}')

df.to_hdf("../../data/data_wo_text_mining_cat.h5", key = 'category_pruning')

'''
size = 304
Next: country variable pruning
'''
