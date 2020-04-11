#!/usr/bin/python
# -*- coding:utf-8 -*-

"""
EDA Kickstarter_2020-02-13T03_20_04_893Z.
profile reports
Input Filename: test_frequency_score.h5, train_frequency_score.h5.h5
Output Filename: EDA_test_profile.html, EDA_train_profile.html

Copyright (c) 2020 Gloria G. Curto.
http://gloriagcurto.info
"""

import pandas as pd
import pandas_profiling


df = pd.read_hdf("../../data/test_frequency_score.h5")

print(df.shape)

profile = pandas_profiling.ProfileReport(df, title='Test EDA profiling report', html={'style':{'full_width':True}})
profile.to_file(output_file="../../EDA_profiles/EDA_test_profile.html")

df = pd.read_hdf("../../data/train_frequency_score.h5")

print(df.shape)

profile = pandas_profiling.ProfileReport(df, title='Training EDA profiling report', html={'style':{'full_width':True}})

profile.to_file(output_file="../../EDA_profiles/EDA_train_profile.html")