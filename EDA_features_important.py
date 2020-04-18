#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Pre-processing of Kickstarter_2020-02-13T03_20_04_893Z.
plots EDA feature importance: profile, spotlight, profile, usd_goal
Input Filename: train, test
Output Filename: plots

Copyright (c) 2020 Gloria G. Curto.
http://gloriagcurto.info
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_count_by_state(df, var_to_plot, palette_dict, title, xlab, ylab, output_path):
    output_filename = output_path + var_to_plot + "_state.pdf"
    fig = plt.figure(figsize=(20,5))
    g = sns.countplot(x=var_to_plot, hue='state_code', data=df, palette=palette_dict)
    g.set_title(f' {title}')
    g.set_xlabel(xlab)
    g.set_ylabel(ylab)
    plt.legend(title='Project state', loc='upper center')
    plt.savefig(output_filename)
    plt.close(fig)

def plot_violin_by_state(df, var_to_plot, palette_dict, title, xlab, ylab, output_path):
    output_filename = output_path + var_to_plot + "_state.pdf"
    fig = plt.figure(figsize=(20,5))
    g = sns.violinplot(x='state_code', y=var_to_plot, hue='state_code', data=df, palette=palette_dict, legend='brief')
    g.set_title(f' {title}')
    g.set_xlabel(xlab)
    g.set_ylabel(ylab)
    plt.legend(title='Project state', loc='upper center')
    plt.savefig(output_filename)
    plt.close(fig)

def plot_boxplot_by_state(df, var_to_plot, palette_dict, title, xlab, ylab, output_path, ylim):
    output_filename = output_path + var_to_plot + "_state.pdf"
    fig = plt.figure(figsize=(20,5))
    g = sns.violinplot(x='state_code', y=var_to_plot, hue='state_code', data=df, palette=palette_dict, legend='brief')
    g.set_title(f' {title}')
    g.set_xlabel(xlab)
    g.set_ylabel(ylab)    
    plt.ylim(ylim)
    plt.legend(title='Project state', loc='upper center')
    plt.savefig(output_filename)
    plt.close(fig)

'''
def counts_dfs (df, variable_name):
    df_state = df.loc[:,[variable_name, 'state_code']]
    print(f'Dimensions: {df_state.shape}')
    
    #Total projects
    df_counts = df_state[variable_name].value_counts().sort_index().reset_index()
    df_counts.columns = [str(variable_name), 'project_count']

    #Successful projects
    df_success = df_state.loc[df_state['state_code'] == 1]
    print(f'Dimensions success : {df_success.shape}')
    df_success_counts = df_success[variable_name].value_counts().sort_index().reset_index()
    df_success_counts.columns = [str(variable_name),'project_count']

    #Failed projects
    df_failed = df_state.loc[df_state['state_code'] == 0]
    print(f'Dimensions failed : {df_failed.shape}')
    df_failed_counts = df_failed[variable_name].value_counts().sort_index().reset_index()
    df_failed_counts.columns = [str(variable_name),'project_count']

    return df_counts, df_success_counts, df_failed_counts
'''


train = pd.read_hdf("../../data/train_frequency_score.h5")
test = pd.read_hdf("../../data/test_frequency_score.h5")
df = pd.concat([test, train], axis=0)
print(df.head(3))

#function counts df
#df_counts, df_success_counts, df_failed_counts = counts_dfs (df, 'spotlight')

# plot counts by spotlight and state
output_path = "../../results/model/important_features_EDA/spotlight/" 
var_to_plot = 'spotlight'
title = 'Spotlight and project state'
xlab='spotlight'
ylab='Project count'
palette_dict = {1:'#66cdaa', 0: '#ff8b61'}

plot_count_by_state(df, var_to_plot, palette_dict, title, xlab, ylab, output_path)

# plot counts by staff_pick and state
output_path = "../../results/model/important_features_EDA/staff_pick/" 
var_to_plot = 'staff_pick'
title = 'staff pick and project state'
xlab='staff pick'
ylab='Project count'
palette_dict = {1:'#66cdaa', 0: '#ff8b61'}

plot_count_by_state(df, var_to_plot, palette_dict, title, xlab, ylab, output_path)

# Frequency score
output_path = "../../results/model/important_features_EDA/profile/" 
var_to_plot = 'profile'
title = 'Word frequency score and project state'
xlab='word frequency score'
ylab='Project count'
palette_dict = {1:'#66cdaa', 0: '#ff8b61'}

plot_violin_by_state(df, var_to_plot, palette_dict, title, xlab, ylab, output_path)

# usd_goal
'''
# Set axes style to white

fig = plt.figure(figsize=(20,5))
with sns.axes_style("white"):
    g = sns.swarmplot(x="state_code", y="usd_goal", data=df, palette=palette_dict)
    g.set_title('Goal in USD and the state of the project') 
    g.set_xlabel(xlab)
    g.set_ylabel(ylab)
    plt.legend(title='Project state', loc='upper left')
    plt.savefig('../../results/model/important_features_EDA/usd_goal/swarmplot_usd_goal_state.pdf')
    plt.show()
    plt.close(fig)             
'''
output_path = "../../results/model/important_features_EDA/usd_goal/" 
var_to_plot = 'usd_goal'
title = 'Goal (USD) and project state'
xlab='goal (USD)'
ylab='Project count'
palette_dict = {1:'#66cdaa', 0: '#ff8b61'}
ylim=(0,1e8)

plot_boxplot_by_state(df, var_to_plot, palette_dict, title, xlab, ylab, output_path, ylim)

output_path = "../../results/model/important_features_EDA/usd_goal/ylim_" 
ylim = (0, 75000)
plot_boxplot_by_state(df, var_to_plot, palette_dict, title, xlab, ylab, output_path, ylim)

# profile

output_path = "../../results/model/important_features_EDA/profile/" 
var_to_plot = 'profile'
title = 'Profile fill in score and project state'
xlab='Profile fill in score'
ylab='Project count'
palette_dict = {1:'#66cdaa', 0: '#ff8b61'}

plot_violin_by_state(df, var_to_plot, palette_dict, title, xlab, ylab, output_path)

# Correlation matrix
# Compute the correlation matrix
corr = df.corr()

# Set up the matplotlib figure
g, ax = plt.subplots(figsize=(30, 30))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
g = sns.heatmap(corr, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})

g.set_title('Correlation matrix')
plt.tight_layout()
plt.savefig('../../results/model/important_features_EDA/correlation_matrix.pdf')

plt.close()
