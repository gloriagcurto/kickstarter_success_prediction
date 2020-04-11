
#!/usr/bin/python
# -*- coding:utf-8 -*-

"""
Pre-processing of Kickstarter_2020-02-13T03_20_04_893Z.
Remove unnecesary features and split feature and targets
Input Filename: train_frequency_score.h5, train_ts_frequency_score.h5, 
Output Filename: train.h5, train_ts.h5

Copyright (c) 2020 Gloria G. Curto.
http://gloriagcurto.info
"""
import pandas as pd

train = pd.read_hdf("../../data/train_frequency_score.h5")

cols = [col for col in train.columns]

with open('../../data/colnames.txt', "w") as output:
    output.write(str(cols))

print(f'Dimensions before pruning: {train.shape}')

y_train = train['state_code']
print(f'Dimensions of y train: {y_train.shape}')

to_drop = ['state_code','blurb', 'name', 'text']
train.drop(to_drop, axis=1, inplace=True)

print(f'Dimensions of X_train: {train.shape}')

train.to_hdf("../../data/data_model/X_train.h5", key="X_train")
y_train.to_hdf("../../data/data_model/y_train.h5", key="y_train")


# time series
train = pd.read_hdf("../../data/train_ts_frequency_score.h5")

cols = [col for col in train.columns]

with open('../../data/colnames.txt', "w") as output:
    output.write(str(cols))

print(f'Dimensions before pruning: {train.shape}')

y_train = train['state_code']
print(f'Dimensions of y train: {y_train.shape}')

to_drop = ['state_code','blurb', 'name', 'text']
train.drop(to_drop, axis=1, inplace=True)

print(f'Dimensions of X_train: {train.shape}')

train.to_hdf("../../data/data_model/X_train_ts.h5", key="X_train")
y_train.to_hdf("../../data/data_model/y_train_ts.h5", key="y_train")




