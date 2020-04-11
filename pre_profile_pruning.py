#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Pre-processing of Kickstarter_2020-02-13T03_20_04_893Z.
Profile variables pruning
Input Filename: data_nd_wo_text_mining.h5
Output Filename: data_wo_text_mining_profile.h5

Copyright (c) 2020 Gloria G. Curto.
http://gloriagcurto.info
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_hdf('../../data/data_nd_wo_text_mining.h5')
print(df.shape)

print(df.iloc[:, 50:57].head(3))

cols = [col for col in df.columns]
print(cols)

# Profile score accounting for profile completeness
df['profile']= df['profile_state'] + df['profile_name'] + df['profile_blurb'] + df['profile_background_color']+ df['profile_text_color'] + df['profile_link_background_color'] + df['profile_link_text_color'] + df['profile_link_text'] + df['profile_link_url'] + df['profile_show_feature_image'] + df['profile_should_show_feature_image_section']

print(f'Dimensions before pruning: {df.shape}')

df.drop(['profile_state', 'profile_name', 'profile_blurb', 'profile_background_color', 'profile_text_color', 'profile_link_background_color', 'profile_link_text_color', 'profile_link_text', 'profile_link_url', 'profile_show_feature_image', 'profile_should_show_feature_image_section'], axis=1, inplace=True)
print(df.profile.iloc[:5])

print(f'Dimensions after pruning: {df.shape}')

df.to_hdf("../../data/data_wo_text_mining_profile.h5", key = 'profile_pruning')

'''
size: 418
Next: category_name variable pruning
'''