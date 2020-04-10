#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Pre-processing of Kickstarter_2020-02-13T03_20_04_893Z.
Preliminary exploration and pre-procesing of the data decoded from the JSON encoded columns from joined.csv.
Input Filename: deserialized.h5
Output Filename: deserialized_dummied.h5

Copyright (c) 2020 Gloria G. Curto.
http://gloriagcurto.info
'''

import pandas as pd

deserialized = pd.read_hdf("../../data/deserialized.h5")

# Dimensions and missing values

print(f'Dimensions of the deserialized data: {deserialized.shape}')
print('Number of missing values by column:')
for column in deserialized :
    print(column, deserialized.loc[:,column].isnull().sum())

# Visualization of variables:

print(deserialized.iloc[0:7,60:70])

# Column names

print(deserialized.columns)

# Check columns values.
# Total number, size and number of unique values. Count gives an idea of Na if compared to size.

print(deserialized.iloc[:,0:10].agg(['count', 'size', 'nunique']))

print(deserialized.iloc[:,10:20].agg(['count', 'size', 'nunique']))

print(deserialized.iloc[:,20:30].agg(['count', 'size', 'nunique']))

print(deserialized.iloc[:,30:40].agg(['count', 'size', 'nunique']))

print(deserialized.iloc[:,40:50].agg(['count', 'size', 'nunique']))

print(deserialized.iloc[:,50:60].agg(['count', 'size', 'nunique']))

print(deserialized.iloc[:,60:70].agg(['count', 'size', 'nunique']))

# After visualization of the column contents and evaluation of number of unique values and missing values, drop the following columns because they might not be informative enough. Serveral columns to drop include urls.

columns_to_drop = ['category_position', 'category_parent_id', 'category_id', 'category_color', 'category_urls.web.discover','creator_id',
                   'creator_name','creator_slug', 'creator_is_registered', 'creator_chosen_currency',
                   'creator_is_superbacker', 'creator_avatar.thumb', 'creator_avatar.small', 'creator_avatar.medium',
                   'creator_urls.web.user', 'creator_urls.api.user', 'location_id', 'location_name', 'location_slug',
                   'location_short_name', 'location_displayable_name', 'location_state', 'location_type', 'location_is_root',
                   'location_urls.web.discover', 'location_urls.web.location', 'location_urls.api.nearby_projects', 'photo_key',
                   'photo_full', 'photo_ed', 'photo_med', 'photo_little', 'photo_small', 'photo_thumb', 'photo_1024x576', 'photo_1536x864',
                    'profile_id', 'profile_project_id', 'profile_state_changed_at', 'profile_background_image_attributes.id', 'profile_background_image_attributes.image_urls.default', 'profile_background_image_attributes.image_urls.baseball_card',
                 'profile_feature_image_attributes.id', 'urls_web.project', 'urls_web.rewards', 'urls_api.star', 'urls_api.message_creator', 'urls_web.message_creator']

deserialized.drop(columns_to_drop, axis=1, inplace=True)

print(f'Deserialized dimensions after column drop: {deserialized.shape}')

print(deserialized.head())

# Pre-processing of variable values

# category_slug contains partial redundant information with category_name and category_parent_name. Drop category_slug

print(deserialized.iloc[:30, 0:3])

deserialized.drop(['category_slug'], axis=1, inplace=True)

print(f'Deserialized dimensions after category_slug column drop: {deserialized.shape}')

# Fill category_parent_name missing values with their corresponding category_name value.

print(deserialized.iloc[:,0:2].agg(['count', 'size', 'nunique']))

print(deserialized.iloc[:,0:2].isnull().sum())

print(deserialized.iloc[:,1].fillna(value=deserialized.iloc[:,0], axis=0, inplace=True))
print(deserialized.iloc[:,0:2].head(10))

# Exploration of location related variables location_localized_name	location_country	location_expanded_country

print(deserialized.iloc[:,2:5].agg(['count', 'size', 'nunique']))

print(deserialized.iloc[:,2:5].isnull().sum())

print(deserialized.iloc[:,2:5].head(10))

# location_localized_name contains no missing values and 12977. Such a number of levels for a categorical variable might not be informative. Drop location_localized_name.

# location_country and location_expanded_country are redundant. Drop location_country because is less human readable.

deserialized.drop(['location_localized_name', 'location_country'], axis=1, inplace=True)

print(f'Deserialized dimensions after location_localized_name and location_country column drop: {deserialized.shape}')

# Profile related variables.
# Codification into a binary code.

print(deserialized.iloc[:,3:15].columns)
print(deserialized.iloc[:,3:15].head())

# Re-codification of categorical binary variables:
# profile_state: "inactive"=0, "active"=1
# profile_show_feature_image: False=0, True=1
# profile_should_show_feature_image_section: False=0, True=1

deserialized["profile_state"].replace(["inactive","active"],[0, 1], inplace=True)

deserialized.replace([False,True],[0, 1], inplace=True)

print(deserialized.iloc[:,3:15].head())

# The rest of the profile related columns are going to be coded in a binary choice variable (0= missing value, 1= variable contains a project creator provided value)
# ['profile_name', 'profile_blurb', 'profile_background_color', 'profile_text_color', 'profile_link_background_color', 'profile_link_text_color', 'profile_link_text', 'profile_link_url']
#  'profile_background_image_opacity', 'profile_feature_image_attributes.image_urls.default', 'profile_feature_image_attributes.image_urls.baseball_card',
#                  'profile_background_image_attributes.id', 'profile_background_image_attributes.image_urls.default',
#                  'profile_background_image_attributes.image_urls.baseball_card', 'profile_feature_image_attributes.id',
profile_var = ['profile_name', 'profile_blurb', 'profile_background_color', 'profile_text_color',
                  'profile_link_background_color', 'profile_link_text_color', 'profile_link_text', 'profile_link_url']

for column in profile_var :
    deserialized.loc[:, column].replace([None,""],[0, 0], inplace=True)
    deserialized.loc[:, column].mask(deserialized.loc[:, column].ne(0), 1, inplace=True)


print(deserialized.iloc[:,3:15].head())
print(deserialized.iloc[:,20:25].head())

# One hot encoding of categorical variables.
# Keep the original columns for dataviz.
deserialized['category_name_ori'] = deserialized['category_name']
deserialized['category_parent_name_ori'] = deserialized['category_parent_name']
deserialized['location_expanded_country_ori'] = deserialized['location_expanded_country']
# Columns to recode:
# Index(['category_name', 'category_parent_name', 'location_expanded_country'],

deserialized_dummied = pd.get_dummies(deserialized, prefix_sep='_', columns=['category_name', 'category_parent_name', 'location_expanded_country'], drop_first=True)

print(f'New column names: {deserialized_dummied.columns}')
print(f'New dimmensions: {deserialized_dummied.shape}')
print(deserialized_dummied.head())

#save in hdf5 format
#remember to drop columns=['category_name', 'category_parent_name', 'location_expanded_country'] in features matrix

deserialized_dummied.to_hdf("../../data/deserialized_dummied.h5", key = 'deserialized_dummied')

