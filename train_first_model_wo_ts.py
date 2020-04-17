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

def plot_model_interpretation_xgb (model, output_path, file_prefix):
    '''
    xgboost model interpretation plots 
    '''
    
    output_filename = output_path + file_prefix + "_tree.pdf"
    fig = plt.figure(figsize=(100,400))
    xgb.plot_tree(model)
    plt.title("Model decision tree")
    plt.savefig(output_filename)
    plt.close(fig)
    

    output_filename = output_path + file_prefix + "_importance_weight_plot.pdf"
    fig = plt.figure(figsize=(500,8000))
    xgb.plot_importance(model, height=1, max_num_features=15)
    plt.title('importance_type="weight"')
    plt.tight_layout()
    plt.savefig(output_filename)
    plt.close(fig)

    output_filename = output_path + file_prefix + "_importance_cover_plot.pdf"
    fig = plt.figure(figsize=(500,8000))
    xgb.plot_importance(model, height=1, max_num_features=15, importance_type="cover")
    plt.title('importance_type="cover"')
    plt.tight_layout()
    plt.savefig(output_filename)
    plt.close(fig)

    output_filename = output_path + file_prefix + "_importance_gain_plot.pdf"
    fig = plt.figure(figsize=(100,400))
    xgb.plot_importance(model, height=1, max_num_features=15, importance_type="gain")
    plt.title('importance_type="gain"')
    plt.tight_layout()
    plt.savefig(output_filename)
    plt.close(fig)

def eval_metrics (y_true, y_pred,output_path, file_prefix):

    #Printing the classification report
    output_filename = output_path + file_prefix + "_class_report.tex"
    report = classification_report(y_true, y_pred, output_dict=True)
    print(report)
    pd.DataFrame(report).transpose().to_latex(buf=output_filename)

    #confusion matrix
    cm= confusion_matrix(y_true, y_pred)
    print(f'Confusion matrix: {cm}')
    output_filename = output_path + file_prefix + "_confusion_matrix.tex"
    pd.DataFrame(cm).to_latex(buf=output_filename)

plt.close("all")

X_train = pd.read_hdf('../../data/data_model/X_train.h5')
y_train = pd.read_hdf('../../data/data_model/y_train.h5')

X_test =  pd.read_hdf('../../data/data_model/X_test.h5')
y_test =  pd.read_hdf('../../data/data_model/y_test.h5')

#Initializing an XGBClassifier with default parameters and fitting the training data

xgbcl1 = xgb.XGBClassifier(objective='binary:logistic').fit(X_train, y_train)

#Predicting for training set
pred_train_1 = xgbcl1.predict(X_train)

#Metrics for train set
eval_metrics (y_train, pred_train_1,"../../results/model/eval_metrics/", 'xgbcl1_wo_ts_train')

#Predicting for training set
pred_test_1 = xgbcl1.predict(X_test)

#Metrics for train set
eval_metrics (y_test, pred_test_1,"../../results/model/eval_metrics/", 'xgbcl1_wo_ts_test')

# xgb model interpretation
plot_model_interpretation_xgb (xgbcl1, '../../results/model/xgb_plots/', 'xgbcl1')

#Save model

xgbcl1.save_model('../../results/model/xgb_binLog_cl1_wo_ts.model')

# Shuffle labels to test the model

from sklearn.utils import shuffle
y_train_s = shuffle(y_train)

#Initializing an XGBClassifier with default parameters and fitting the training data

xgbcl_s = xgb.XGBClassifier(objective='binary:logistic').fit(X_train, y_train_s)

#Predicting for training set
pred_train_s = xgbcl_s.predict(X_train)

#Metrics for train set
eval_metrics (y_train_s, pred_train_s,"../../results/model/eval_metrics/", 'xgbcl_shuffled_train')

#Predicting for training set
pred_test_s = xgbcl_s.predict(X_test)

#Metrics for train set
eval_metrics (y_test, pred_test_s,"../../results/model/eval_metrics/", 'xgbcl_shuffle_test')

'''
remove spotlight and train the model again
'''