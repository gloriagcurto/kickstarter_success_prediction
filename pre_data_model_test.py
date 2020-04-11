
#!/usr/bin/python
# -*- coding:utf-8 -*-

"""
Pre-processing of Kickstarter_2020-02-13T03_20_04_893Z.
Remove unnecesary features and split feature and targets for test data
Input Filename: test_frequency_score.h5, test_frequency_score.h5, 
Output Filename: test.h5, test_ts.h5

Copyright (c) 2020 Gloria G. Curto.
http://gloriagcurto.info
"""
import pandas as pd

test = pd.read_hdf("../../data/test_frequency_score.h5")

cols = [col for col in test.columns]

with open('../../data/colnames.txt', "w") as output:
    output.write(str(cols))

print(f'Dimensions before pruning: {test.shape}')

y_test = test['state_code']
print(f'Dimensions of y test: {y_test.shape}')

to_drop = ['state_code','blurb', 'name', 'text']
test.drop(to_drop, axis=1, inplace=True)

print(f'Dimensions of X_test: {test.shape}')

test.to_hdf("../../data/data_model/X_test.h5", key="X_test")
y_test.to_hdf("../../data/data_model/y_test.h5", key="y_test")


# time series
test = pd.read_hdf("../../data/test_ts_frequency_score.h5")

cols = [col for col in test.columns]

with open('../../data/colnames.txt', "w") as output:
    output.write(str(cols))

print(f'Dimensions before pruning: {test.shape}')

y_test = test['state_code']
print(f'Dimensions of y test: {y_test.shape}')

to_drop = ['state_code','blurb', 'name', 'text']
test.drop(to_drop, axis=1, inplace=True)

print(f'Dimensions of X_test: {test.shape}')

test.to_hdf("../../data/data_model/X_test_ts.h5", key="X_test")
y_test.to_hdf("../../data/data_model/y_test_ts.h5", key="y_test")