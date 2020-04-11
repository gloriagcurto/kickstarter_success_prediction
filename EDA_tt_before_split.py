#!/usr/bin/python
# -*- coding:utf-8 -*-

"""
EDA Kickstarter_2020-02-13T03_20_04_893Z.
profile reports
Input Filename: 
Output Filename: EDA_tt_ts_split_profile.html

Copyright (c) 2020 Gloria G. Curto.
http://gloriagcurto.info
"""

import pandas as pd
import pandas_profiling


df = pd.read_hdf("../../data/test_training_wo_text_mining.h5")
df.drop(['state_changed_at', 'state_grouped', 'year_state_changed_at'], axis=1, inplace=True)

print(df.shape)

profile = pandas_profiling.ProfileReport(df, title='Test and training before splitting EDA profiling report', html={'style':{'full_width':True}})
profile.to_file(output_file="../../EDA_profiles/EDA_tt_before_split_profile.html")