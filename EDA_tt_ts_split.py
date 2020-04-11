#!/usr/bin/python
# -*- coding:utf-8 -*-

"""
EDA Kickstarter_2020-02-13T03_20_04_893Z.
profile reports
Input Filename: test_ts_frequency_score.h5, train_ts_frequency_score.h5
Output Filename: EDA_ts_test_profile.html, EDA_ts_train_profile.html

Copyright (c) 2020 Gloria G. Curto.
http://gloriagcurto.info
"""

import pandas as pd
import pandas_profiling


df = pd.read_hdf("../../data/test_ts_frequency_score.h5")

print(df.shape)

profile = pandas_profiling.ProfileReport(df, title='Test time series split EDA profiling report', html={'style':{'full_width':True}})
profile.to_file(output_file="../../EDA_profiles/EDA_ts_test_profile.html")

df = pd.read_hdf("../../data/train_ts_frequency_score.h5")

print(df.shape)

profile = pandas_profiling.ProfileReport(df, title='Training time series split EDA profiling report', html={'style':{'full_width':True}})

profile.to_file(output_file="../../EDA_profiles/EDA_ts_train_profile.html")