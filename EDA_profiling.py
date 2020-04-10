#!/usr/bin/python
# -*- coding:utf-8 -*-

"""
EDA Kickstarter_2020-02-13T03_20_04_893Z.
profile reports
Input Filename: joined.csv, train, test 
Output Filename: raw_profile, train_profile, test_profile

Copyright (c) 2020 Gloria G. Curto.
http://gloriagcurto.info
"""

import pandas as pd
import pandas_profiling


df_raw = pd.read_csv('../../data/joined.csv')

print(df_raw.head())
print(df_raw.shape)
raw_profile = pandas_profiling.ProfileReport(df_raw, title='Raw data EDA profiling report', html={'style':{'full_width':True}})
raw_profile.to_file(output_file="../../EDA_profiles/EDA_raw_profile.html")