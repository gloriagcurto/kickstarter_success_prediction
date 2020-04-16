#!/usr/bin/python
# -*- coding:utf-8 -*-

"""
XGBoost traiing Kickstarter_2020-02-13T03_20_04_893Z.
Model training, binary classification, without variable spotlight
Input Filename: train, test
Output Filename: 

Copyright (c) 2020 Gloria G. Curto.
http://gloriagcurto.info
"""

import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix
#from bayes_opt import BayesianOptimization

plt.close("all")

X_train = pd.read_hdf('../../data/data_model/X_train.h5')
X_train.drop( 'spotlight', axis=1, inplace=True)
y_train = pd.read_hdf('../../data/data_model/y_train.h5')

X_test =  pd.read_hdf('../../data/data_model/X_test.h5')
X_test.drop( 'spotlight', axis=1, inplace=True)
y_test =  pd.read_hdf('../../data/data_model/y_test.h5')



#Initializing an XGBClassifier with default parameters and fitting the training data

xgbcl_wo_spot = xgb.XGBClassifier(objective='binary:logistic').fit(X_train, y_train)

#Predicting for training set
pred_train_1 = xgbcl_wo_spot.predict(X_train)

#Printing the classification report

print(classification_report(y_train, pred_train_1))

#Accuracy obtained on the training set
cm = confusion_matrix(y_train, pred_train_1)
acc = cm.diagonal().sum()/cm.sum()
print(acc)

#Predicting for test
pred_test_1 = xgbcl_wo_spot.predict(X_test)


#Printing the classification report

print(classification_report(y_test, pred_test_1))

#Accuracy obtained on the training set
cm_t= confusion_matrix(y_test, pred_test_1)
acc_t = cm_t.diagonal().sum()/cm_t.sum()
print(acc_t)
print(f'Confusion matrix test: {cm_t}')

#
xgb.plot_tree(xgbcl_wo_spot)
plt.show()

# Plot the feature importances
xgb.plot_importance(xgbcl_wo_spot)
plt.show()

xgbcl_wo_spot.save_model('../../results/model/xgb_binLog_cl_wo_spot_wo_ts.model')

