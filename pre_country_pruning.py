#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Pre-processing of Kickstarter_2020-02-13T03_20_04_893Z.
country variables pruning
Input Filename: data_wo_text_mining_cat.h5
Output Filename: data_wo_text_mining_country.h5

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


df = pd.read_hdf('../../data/data_wo_text_mining_cat.h5')

print(df.head(3))

# location_expanded_country_ori

df_state_country = df.loc[:,['location_expanded_country_ori', 'state_grouped']]
print(f'Dimensions: {df_state_country.shape}')

#Total projects
df_counts = df_state_country['location_expanded_country_ori'].value_counts().sort_index().reset_index()
df_counts.columns = ['country','project_count']
print(f'Total projects: {df_counts.head}')

#Successful projects
df_success = df_state_country.loc[df_state_country['state_grouped'] == 'successful']
print(f'Dimensions success : {df_success.shape}')
df_success_counts = df_success['location_expanded_country_ori'].value_counts().sort_index().reset_index()
df_success_counts.columns = ['country','project_count']

#Failed projects
df_failed = df_state_country.loc[df_state_country['state_grouped'] == 'failed']
print(f'Dimensions failed : {df_failed.shape}')
df_failed_counts = df_failed['location_expanded_country_ori'].value_counts().sort_index().reset_index()
df_failed_counts.columns = ['country','project_count']


# Countries with more than 200 projects
df_counts_200 = df_counts.loc[df_counts.project_count >= 200]
print(f'Countries with more than 200 projects : {df_counts_200.shape[0]}')
print(f'Countries: {df_counts.shape[0]}')

df_success_counts_200 = df_success_counts.loc[df_success_counts.project_count >= 200]
print(f'Countries with more than 200 success projects : {df_success_counts_200.shape[0]}')
print(f'Countries: {df_counts.shape[0]}')

df_failed_counts_200 = df_failed_counts.loc[df_failed_counts.project_count >= 200]
print(f'Countries with more than 200 failed projects : {df_failed_counts_200.shape[0]}')
print(f'Countries: {df_counts.shape[0]}')

'''
# plot total projects
plt.figure(figsize=(50,30))
ax1 = plt.subplot(311)
plt.plot('country', 'project_count', '-k', data=df_counts)
plt.setp(ax1.get_xticklabels(), fontsize=6)

# successful projects
ax2 = plt.subplot(312, sharex=ax1)
plt.plot('country', 'project_count', '-.g', data=df_success_counts)
# make these tick labels invisible
plt.setp(ax2.get_xticklabels(), visible=False)

# failed projects
ax3 = plt.subplot(313, sharex=ax1, sharey=ax1)
plt.plot('country', 'project_count', ':r', data=df_failed_counts)
plt.savefig("../../images/variable_pruning_EDA/country/nprojects_country_state.svg", format="svg")
plt.show()


# total
plt.figure(figsize=(30,13))
ax1 = plt.subplot(311)
plt.scatter(df_counts_200.country, df_counts_200.project_count, c='k')
plt.setp(ax1.get_xticklabels(), fontsize=12)

# successful
ax2 = plt.subplot(312)
plt.scatter(df_success_counts_200.country, df_success_counts_200.project_count, c='g')
# make these tick labels invisible
plt.setp(ax2.get_xticklabels(), fontsize=12)

# failed
ax3 = plt.subplot(313, sharex=ax1, sharey=ax1)
plt.scatter(df_failed_counts_200.country, df_failed_counts_200.project_count, c='r')
plt.setp(ax3.get_xticklabels(), fontsize=12)
plt.savefig("../../images/variable_pruning_EDA/country/nprojects_scatter_country_state_200.svg", format="svg")
plt.show()

'''

#plot and compute percentages of successful projects by secondary category to decide one hot encoded columns under category_name to drop.

# plot counts by categories and state
output_path = "../../images/variable_pruning_EDA/country/nprojects_" 
var_to_plot = 'location_expanded_country_ori'
title = 'Country'
xlab=''
ylab='Project count'
palette_dict = dict(successful='#66cdaa', failed='#ff8b61')

for country in df.location_expanded_country_ori.unique():
    df_sub = df[df['location_expanded_country_ori']==country]
    #print(country)
    plot_count_by_state(df_sub, var_to_plot, country, palette_dict, title, xlab, ylab, output_path)

# Percentage success

percen = []
countries = []
for country in df.location_expanded_country_ori.unique() : 
    df_sub = df[df['location_expanded_country_ori']==country]
    countries.append(country)
    percen.append((df_sub[df_sub['state_grouped']=="successful"].size/df_sub.size)*100)

p_success = pd.DataFrame({'countries': countries, 'percentages':percen})
p_success.to_csv('../../var_pruning_notebooks/p_success_country.csv')

print(p_success.head(10))

print(f'Percentage of success: {(df[df.state_grouped=="successful"].size/df.size)*100}')
print(f'Percentage of failure: {(df[df.state_grouped=="failed"].size/df.size)*100}')

# Keep countries with a percentage of successful projects of at least 55% 
keep_p = p_success[p_success.percentages >= 55]
print(f'Number of countries with more than 55% of successful projects: {len(keep_p)}')
keep = keep_p.merge(right=df_counts, how='inner', left_on='countries', right_on='country')
keep.to_csv('../../data/keep_countries_55success.csv')
'''
Keep countries with more than 55% of successful projects ands more than 50 projects:
countries:
United Kingdom, Hong Kong, Japan, Singapore, China, Poland, Israel, Taiwan, Czech Republic, Greece, Indonesia, Argentina, Kenya, Iceland, Ghana, Portugal, Slovenia, Finland

'''
cols = [col for col in df.columns]
#print(cols)

with open('../../data/colnames.txt', "w") as output:
    output.write(str(cols))

#change based on success
to_drop = ['location_expanded_country_Aland Islands', 'location_expanded_country_Albania', 'location_expanded_country_Algeria', 'location_expanded_country_Angola', 'location_expanded_country_Anguilla', 'location_expanded_country_Antarctica', 'location_expanded_country_Antigua and Barbuda', 'location_expanded_country_Armenia', 'location_expanded_country_Australia', 'location_expanded_country_Austria', 'location_expanded_country_Azerbaijan', 'location_expanded_country_Bahamas', 'location_expanded_country_Bahrain', 'location_expanded_country_Bangladesh', 'location_expanded_country_Barbados', 'location_expanded_country_Belarus', 'location_expanded_country_Belgium', 'location_expanded_country_Belize', 'location_expanded_country_Benin', 'location_expanded_country_Bhutan', 'location_expanded_country_Bolivia', 'location_expanded_country_Bosnia and Herzegovina', 'location_expanded_country_Botswana', 'location_expanded_country_Brazil', 'location_expanded_country_Bulgaria', 'location_expanded_country_Burkina Faso', 'location_expanded_country_Cambodia', 'location_expanded_country_Cameroon', 'location_expanded_country_Canada', 'location_expanded_country_Cape Verde', 'location_expanded_country_Chad', 'location_expanded_country_Chile', 'location_expanded_country_Cocos (Keeling) Islands', 'location_expanded_country_Colombia', 'location_expanded_country_Congo', 'location_expanded_country_Congo, the Democratic Republic of', 'location_expanded_country_Cook Islands', 'location_expanded_country_Costa Rica', "location_expanded_country_Cote d'Ivoire", 'location_expanded_country_Croatia', 'location_expanded_country_Cuba', 'location_expanded_country_Curacao', 'location_expanded_country_Cyprus', 'location_expanded_country_Denmark', 'location_expanded_country_Djibouti', 'location_expanded_country_Dominica', 'location_expanded_country_Dominican Republic', 'location_expanded_country_Ecuador', 'location_expanded_country_Egypt', 'location_expanded_country_El Salvador', 'location_expanded_country_Equatorial Guinea', 'location_expanded_country_Estonia', 'location_expanded_country_Ethiopia', 'location_expanded_country_Faroe Islands', 'location_expanded_country_Fiji', 'location_expanded_country_France', 'location_expanded_country_French Guiana', 'location_expanded_country_French Polynesia', 'location_expanded_country_Gabon', 'location_expanded_country_Gambia', 'location_expanded_country_Georgia', 'location_expanded_country_Germany', 'location_expanded_country_Gibraltar', 'location_expanded_country_Greenland', 'location_expanded_country_Guadeloupe', 'location_expanded_country_Guam', 'location_expanded_country_Guatemala', 'location_expanded_country_Guinea', 'location_expanded_country_Guyana', 'location_expanded_country_Haiti', 'location_expanded_country_Honduras', 'location_expanded_country_Hungary', 'location_expanded_country_India', 'location_expanded_country_Iran', 'location_expanded_country_Iraq', 'location_expanded_country_Ireland', 'location_expanded_country_Italy', 'location_expanded_country_Jamaica', 'location_expanded_country_Jordan', 'location_expanded_country_Kazakhstan', 'location_expanded_country_Kiribati', 'location_expanded_country_Kosovo', 'location_expanded_country_Kuwait', 'location_expanded_country_Kyrgyzstan', 'location_expanded_country_Laos', 'location_expanded_country_Latvia', 'location_expanded_country_Lebanon', 'location_expanded_country_Lesotho', 'location_expanded_country_Liberia', 'location_expanded_country_Libyan Arab Jamahiriya', 'location_expanded_country_Lithuania', 'location_expanded_country_Luxembourg', 'location_expanded_country_Macedonia', 'location_expanded_country_Madagascar', 'location_expanded_country_Malawi', 'location_expanded_country_Malaysia', 'location_expanded_country_Maldives', 'location_expanded_country_Mali', 'location_expanded_country_Malta', 'location_expanded_country_Martinique', 'location_expanded_country_Mauritius', 'location_expanded_country_Mexico', 'location_expanded_country_Micronesia', 'location_expanded_country_Moldova', 'location_expanded_country_Monaco', 'location_expanded_country_Mongolia', 'location_expanded_country_Morocco', 'location_expanded_country_Mozambique', 'location_expanded_country_Myanmar', 'location_expanded_country_Namibia', 'location_expanded_country_Nepal', 'location_expanded_country_Netherlands', 'location_expanded_country_New Caledonia', 'location_expanded_country_New Zealand', 'location_expanded_country_Nicaragua', 'location_expanded_country_Niger', 'location_expanded_country_Nigeria', 'location_expanded_country_North Korea', 'location_expanded_country_Norway', 'location_expanded_country_Pakistan', 'location_expanded_country_Palestinian Territories', 'location_expanded_country_Panama', 'location_expanded_country_Papua New Guinea', 'location_expanded_country_Paraguay', 'location_expanded_country_Peru', 'location_expanded_country_Philippines', 'location_expanded_country_Pitcairn', 'location_expanded_country_Puerto Rico', 'location_expanded_country_Qatar', 'location_expanded_country_Reunion', 'location_expanded_country_Romania', 'location_expanded_country_Russia', 'location_expanded_country_Rwanda', 'location_expanded_country_Saint Kitts and Nevis', 'location_expanded_country_Saint Lucia', 'location_expanded_country_Saint Vincent and the Grenadines', 'location_expanded_country_Samoa', 'location_expanded_country_Saudi Arabia', 'location_expanded_country_Senegal', 'location_expanded_country_Serbia', 'location_expanded_country_Seychelles', 'location_expanded_country_Sierra Leone', 'location_expanded_country_Sint Maarten (Dutch part)', 'location_expanded_country_Slovakia', 'location_expanded_country_Somalia', 'location_expanded_country_South Africa', 'location_expanded_country_South Korea', 'location_expanded_country_South Sudan', 'location_expanded_country_Spain', 'location_expanded_country_Sri Lanka', 'location_expanded_country_Sudan', 'location_expanded_country_Suriname', 'location_expanded_country_Svalbard and Jan Mayen', 'location_expanded_country_Swaziland', 'location_expanded_country_Sweden', 'location_expanded_country_Switzerland', 'location_expanded_country_Syrian Arab Republic', 'location_expanded_country_Tajikistan', 'location_expanded_country_Tanzania', 'location_expanded_country_Thailand', 'location_expanded_country_Timor-Leste', 'location_expanded_country_Tonga', 'location_expanded_country_Trinidad and Tobago', 'location_expanded_country_Tunisia', 'location_expanded_country_Turkey', 'location_expanded_country_Turks and Caicos Islands', 'location_expanded_country_Uganda', 'location_expanded_country_Ukraine', 'location_expanded_country_United Arab Emirates', 'location_expanded_country_United States', 'location_expanded_country_Uruguay', 'location_expanded_country_Vanuatu', 'location_expanded_country_Vatican City', 'location_expanded_country_Venezuela', 'location_expanded_country_Viet Nam', 'location_expanded_country_Virgin Islands, U.S.', 'location_expanded_country_Yemen', 'location_expanded_country_Zambia', 'location_expanded_country_Zimbabwe']

print(f'Dimensions before pruning: {df.shape}')

df.drop(to_drop, axis=1, inplace=True)

print(f'Dimensions after pruning: {df.shape}')

df.to_hdf("../../data/data_wo_text_mining_country.h5", key = 'country_pruning')

'''
size: 125
Next: currency variable pruning
'''