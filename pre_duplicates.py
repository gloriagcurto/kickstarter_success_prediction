
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Pre-processing of Kickstarter_2020-02-13T03_20_04_893Z.
Remove duplicates
Input Filename: data_wo_text_mining.h5
Output Filename: data_nd_wo_text_mining.h5

Copyright (c) 2020 Gloria G. Curto.
http://gloriagcurto.info
'''


import pandas as pd
import numpy as np

df = pd.read_hdf('../../data/data_wo_text_mining.h5')

print(f'Dimensions before duplicate drop: {df.shape}')
print(f'Duplicates: {df.duplicated().sum()}')
print(df[df.duplicated()==True].head(5))
df.drop_duplicates(keep='first', inplace=True)

print(f'Dimensions after duplicate drop: {df.shape}')
print(f'Duplicates: {df.duplicated().sum()}')

# Output file

df.to_hdf("../../data/data_nd_wo_text_mining.h5", key = 'data_nd_wo_text_mining')

'''
Next: variable pruning and text mining
'''