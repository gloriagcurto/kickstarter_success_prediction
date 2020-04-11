
#!/usr/bin/python
# -*- coding:utf-8 -*-

"""
Pre-processing of Kickstarter_2020-02-13T03_20_04_893Z.
Remove unnecesary features and split train test (25%)
Input Filename: data_english.h5
Output Filename: train_wo_frequency_score.h5, test_wo_frequency_score.h5

Copyright (c) 2020 Gloria G. Curto.
http://gloriagcurto.info
"""
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_hdf("../../data/data_english.h5")

print(f'Dimensions of English filtered data: {df.shape}')
'''
cols = [col for col in df.columns]

with open('../../data/colnames.txt', "w") as output:
    output.write(str(cols))
'''

print(f'Dimensions before pruning: {df.shape}')

to_drop = ['backers_count','state', 'created_at', 'deadline', 'launched_at', 'id', 'usd_pledged', 'weekday_state_changed_at', 'month_state_changed_at', 'found_rising_duration', 'currency_orig', 'category_name_ori', 'location_expanded_country_ori']

print(f'Dimensions of columns to drop: {len(to_drop)}')

df.drop(to_drop, axis=1, inplace=True)

cols = [col for col in df.columns]
with open('../../data/colnames_training.txt', "w") as output:
    output.write(str(cols))

print(f'Dimensions after pruning: {df.shape}')

df.to_hdf("../../data/test_training_wo_text_mining.h5", key = 'test_training_wo_text_mining')

#Check projects by year (state changed)

''' Results from pre_year_EDA.py
year	percentages	project_count	
2020	61,0364683301344	1563	
2019	67,2209026128266	16840	18403
2018	54,2246282464195	14593	32996
2017	48,8882229547498	16685	
2016	45,8090641815055	18027	
2015	40,9574019401097	23710	
2014	48,7238557138527	16495	
2013	78,1746632359619	6607	
2012	76,7904781448799	4873	
2011	75,4701211867948	2393	
2010	72,0547945205479	730	
2009	75,6756756756757	74	
		                    122590	

If I use 2020 (1563 projects) and 2019 projects (16840 projects) as test data, I will have 15% as test (18403×100÷122590 = 15,011828045% of total projects).
If I use also oct-dec 2018 (26,915735378 %)

I also do a random train test split.
'''
# Temporal series split
#Remove unwanted variables
df.drop('state_changed_at', axis=1, inplace=True)

print(f'Dimensions: {df.shape}')

test_ts = df.loc[df['year_state_changed_at'].isin([2020, 2019, 2018])]
test_ts.drop('year_state_changed_at', axis=1, inplace=True)

print(f'Dimensions of ts test: {test_ts.shape}')

train_ts = df.loc[~df['year_state_changed_at'].isin([2020, 2019, 2018])]
train_ts.drop('year_state_changed_at', axis=1, inplace=True)

train_ts.to_hdf("../../data/train_ts_wo_frequency_score.h5", key="ts_train_wo_fscore")

test_ts.to_hdf("../../data/test_ts_wo_frequency_score.h5", key="ts_test_wo_fscore")

#check subset
print(f'Dimensions of ts train: {train_ts.shape}')

print(f' {test_ts.shape[0] + train_ts.shape[0] == df.shape[0]}')

# Regular split

df.drop('year_state_changed_at', axis=1, inplace=True)

train, test = train_test_split(df, train_size= 0.73,random_state=37)

print(f'Dimensions of training set without frequency score: {train.shape}')
train.to_hdf("../../data/train_wo_frequency_score.h5", key="train_wo_fscore")

print(f'Dimensions of test set without frequency score: {test.shape}')

test.to_hdf("../../data/test_wo_frequency_score.h5", key="test_wo_fscore")

'''
y = df['state_code']
X = df.drop(['state_changed_at', 'year_state_changed_at', 'state_grouped', 'state_code'], axis=1)

X_train, y_train, X_test, y_test = train_test_split(X, y, train_size= 0.2,random_state=37)


print(f'Dimensions of training set without frequency score: {train.shape}')
train.to_hdf("../../data/train_wo_frequency_score.h5", key="train_wo_fscore")

print(f'Dimensions of test set without frequency score: {test.shape}')

test.to_hdf("../../data/test_wo_frequency_score.h5", key="test_wo_fscore")

Next:
EDA profiling
text mining script for frequency score generation in train, later test
remove 'state_changed_at', 'year_state_changed_at' and 'state_grouped', before saving test and train
'state_code' = target
'''