#!/usr/bin/python
# -*- coding:utf-8 -*-

"""
XGBoost traiing Kickstarter_2020-02-13T03_20_04_893Z.
Model training, binary classification, bayesian optimization
Input Filename: train, test , train_ts, test_ts
Output Filename: 

Copyright (c) 2020 Gloria G. Curto.
http://gloriagcurto.info
"""

import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix
from bayes_opt import BayesianOptimization

plt.close("all")

X_train = pd.read_hdf('../../data/data_model/X_train.h5')
y_train = pd.read_hdf('../../data/data_model/y_train.h5')

X_test =  pd.read_hdf('../../data/data_model/X_test.h5')
y_test =  pd.read_hdf('../../data/data_model/y_test.h5')


#Converting the dataframe into XGBoost’s Dmatrix object
dtrain = xgb.DMatrix(data=X_train, label=y_train)
dtest = xgb.DMatrix(data=X_test, label=y_test)

#Initializing an XGBClassifier with default parameters and fitting the training data

xgbcl1 = xgb.XGBClassifier(objective='binary:logistic').fit(X_train, y_train)

#Predicting for training set
pred_train_1 = xgbcl1.predict(X_train)

#Printing the classification report

print(classification_report(y_train, pred_train_1))

#Accuracy obtained on the training set
cm = confusion_matrix(y_train, pred_train_1)
acc = cm.diagonal().sum()/cm.sum()
print(acc)

#Predicting for test
pred_test_1 = xgbcl1.predict(X_test)


#Printing the classification report

print(classification_report(y_test, pred_test_1))

#Accuracy obtained on the training set
cm_t= confusion_matrix(y_test, pred_test_1)
acc_t = cm_t.diagonal().sum()/cm_t.sum()
print(acc_t)
print(f'Confusion matrix test: {cm_t}')

#
xgb.plot_tree(xgbcl1)
plt.show()

# Plot the feature importances
xgb.plot_importance(xgbcl1)
plt.show()
'''
#Bayesian Optimization function for xgboost
#specify the parameters you want to tune as keyword arguments
def bo_tune_xgb(max_depth, gamma, n_estimators ,learning_rate):
    params = {'objective': 'binary:logistic',
              'max_depth': int(max_depth),
              'gamma': gamma,
              'n_estimators': int(n_estimators),
              'learning_rate':learning_rate,
              'subsample': 0.8,
              'eval_metric': 'auc'}

    #Cross validating with the specified parameters in 5 folds and 70 iterations
    cv_results = xgb.cv(params, dtrain,  num_boost_round=70, nfold=5, early_stopping_rounds=100, as_pandas=True,  seed=37)
    return cv_results["test-auc-mean"].iloc[-1]

#Invoking the Bayesian Optimizer with the specified parameters to tune
xgb_bo = BayesianOptimization(bo_tune_xgb, { 'max_depth': (3, 10),
                                             'gamma': (0, 5),
                                             'learning_rate':(0, 1),
                                             'n_estimators':(100, 500)
                                            })
#print(xgb_bo.columns)

#performing Bayesian optimization for 5 iterations with 8 steps of random exploration with an #acquisition function of expected improvement
xgb_bo.maximize(n_iter=5, init_points=8, acq='ei')

#Extracting the best parameters
params = xgb_bo.max['params']
print(params)

#Converting the max_depth and n_estimator values from float to int
params['max_depth']= int(params['max_depth'])
params['n_estimators']= int(params['n_estimators'])

#Initialize an XGBClassifier with the tuned parameters and fit the training data

xgbcl_bo = xgb.XGBClassifier(**params ).fit(X_train, y_train)

#predicting for training set
pred_train_bo = xgbcl_bo.predict(X_train)

#Looking at the classification report
print(classification_report(y_train, pred_train_bo))

#Attained prediction accuracy on the training set
cm_bo = confusion_matrix(y_train, pred_train_bo)
print(cm_bo)

#predicting for test set
pred_test_bo = xgbcl_bo.predict(X_test)

#Looking at the classification report
print(classification_report(y_test, pred_test_bo))

#Attained prediction accuracy on the training set
cm_t_bo = confusion_matrix(y_test, pred_test_bo)
print(cm_t_bo)


#Save model

xgbcl1.save_model('../../results/model/xgb_binLog_cl1_bay_opt_wo_ts.model')


# Shuffle labels to test the model


from sklearn.utils import shuffle
y_train_s = shuffle(y_train)

#Initializing an XGBClassifier with default parameters and fitting the training data

xgbcl_s = xgb.XGBClassifier(objective='binary:logistic').fit(X_train, y_train_s)
#Predicting for training set
pred_train_s = xgbcl_s.predict(X_train)


#Printing the classification report

print(classification_report(y_train_s, pred_train_s))

#Accuracy obtained on the training set
cm_s = confusion_matrix(y_train_s, pred_train_s)
acc_s = cm_s.diagonal().sum()/cm_s.sum()
print(acc_s)

#Predicting for test
pred_test_s = xgbcl_s.predict(X_test)


#Printing the classification report

print(classification_report(y_test, pred_test_s))

#Accuracy obtained on the training set
cm_t_s= confusion_matrix(y_test, pred_test_s)
acc_t_s = cm_t_s.diagonal().sum()/cm_t_s.sum()
print(acc_t_s)
print(f'Confusion matrix test: {cm_t_s}')



# dump model
xgbcl1.dump_model('../../results/model/dump.raw.txt')
# dump model with feature map
xgbcl1.dump_model('../../results/model/dump.raw.txt', '../../results/model/featmap.txt')


#A saved model can be loaded as follows:

bst = xgb.Booster({'nthread': 4})  # init model
bst.load_model('model.bin')  # load data
'''