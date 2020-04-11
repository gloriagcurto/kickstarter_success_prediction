
#!/usr/bin/python
# -*- coding:utf-8 -*-

"""
Pre-processing of Kickstarter_2020-02-13T03_20_04_893Z.
Remove unnecesary features and split train test (25%)
Input Filename: data_english.h5
Output Filename:*.pdf, *.csv

Copyright (c) 2020 Gloria G. Curto.
http://gloriagcurto.info
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_count_by_state(df, var_to_plot, cat, palette_dict, title, xlab, ylab, output_path):
    output_filename = output_path + str(cat) + "_state.pdf"
    fig = plt.figure(figsize=(20,5))
    g = sns.countplot(x=var_to_plot, hue='state_grouped', data=df_sub, palette=palette_dict)
    g.set_title(f' {title} {cat}')
    g.set_xlabel(xlab)
    g.set_ylabel(ylab)
    plt.legend(title='Project state', loc='upper left')
    plt.savefig(output_filename)
    plt.close(fig)

df = pd.read_hdf("../../data/test_training_wo_text_mining.h5")

print(f'Dimensions of data: {df.shape}')

df_state_year = df.loc[:,['year_state_changed_at', 'state_grouped']]
print(f'Dimensions: {df_state_year.shape}')

#Total projects
df_counts = df_state_year['year_state_changed_at'].value_counts().sort_index().reset_index()
df_counts.columns = ['year','project_count']

#plot and compute percentages of successful projects by secondary category to decide one hot encoded columns under category_name to drop.

# plot counts by categories and state
output_path = "../../images/variable_pruning_EDA/year/nprojects_" 
var_to_plot =  'year_state_changed_at'
title = 'Year project state changed'
xlab=''
ylab='Project count'
palette_dict = dict(successful='#66cdaa', failed='#ff8b61')

for year in df.year_state_changed_at.unique():
    df_sub = df[df['year_state_changed_at']==year]
    #print(year)
    plot_count_by_state(df_sub, var_to_plot, year, palette_dict, title, xlab, ylab, output_path)

# Percentage success

percen = []
years = []
for year in df.year_state_changed_at.unique() : 
    df_sub = df[df['year_state_changed_at']==year]
    years.append(year)
    percen.append((df_sub[df_sub['state_grouped']=="successful"].size/df_sub.size)*100)

p_success = pd.DataFrame({'years': years, 'percentages':percen})
p_success.to_csv('../../var_pruning_notebooks/p_success_year.csv')

print(p_success.head(10))

print(f'Percentage of success: {(df[df.state_grouped=="successful"].size/df.size)*100}')
print(f'Percentage of failure: {(df[df.state_grouped=="failed"].size/df.size)*100}')

# Keep years with a percentage of successful projects of at least 55% 
keep_p = p_success[p_success.percentages >= 55]
print(f'Number of years with more than 55% of successful projects: {len(keep_p)}')

keep = p_success.merge(right=df_counts, how='inner', left_on='years', right_on='year')
keep.to_csv('../../data/years_success.csv')

'''
year	percentages	project_count	
2020	61,0364683301344	1563	
2019	67,2209026128266	16840	18403
2018	54,2246282464195	14593	
2017	48,8882229547498	16685	
2016	45,8090641815055	18027	
2015	40,9574019401097	23710	
2014	48,7238557138527	16495	
2013	78,1746632359619	6607	
2012	76,7904781448799	4873	
2011	75,4701211867948	2393	
2010	72,0547945205479	730	
2009	75,6756756756757	74	
		                    122590	



If I use 2020 (1563 projects) and 2019 projects (16840 projects) as test data, I will have 15% as test (18403×100÷122590 = 15,011828045% of total projects).
If I use also oct-dec 2018 ()

I also do a random train test split.
'''

#Remove unwanted variables