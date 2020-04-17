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
from bayes_opt import BayesianOptimization



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
    print()
    pd.DataFrame(report).transpose().to_latex(buf=output_filename)

    #confusion matrix
    cm= confusion_matrix(y_true, y_pred)
    print(f'Confusion matrix: {cm}')
    output_filename = output_path + file_prefix + "_confusion_matrix.tex"
    pd.DataFrame(cm).to_latex(buf=output_filename)

plt.close("all")

# Read train and test files and drop the variable spotlight due to its high correlation to the target

X_train = pd.read_hdf('../../data/data_model/X_train.h5')
X_train.drop( 'spotlight', axis=1, inplace=True)
y_train = pd.read_hdf('../../data/data_model/y_train.h5')

X_test =  pd.read_hdf('../../data/data_model/X_test.h5')
X_test.drop( 'spotlight', axis=1, inplace=True)
y_test =  pd.read_hdf('../../data/data_model/y_test.h5')

dtrain = xgb.DMatrix(data=X_train, label=y_train)
dtest = xgb.DMatrix(data=X_test, label=y_test)


#Initializing an XGBClassifier with default parameters and fitting the training data

xgbcl_wo_spot = xgb.XGBClassifier(objective='binary:logistic').fit(X_train, y_train)

#Predicting for training set
pred_train_1 = xgbcl_wo_spot.predict(X_train)

#Metrics for train set
eval_metrics (y_train, pred_train_1,"../../results/model/eval_metrics/", 'xgbcl_wo_spot_train')

#Predicting for test
pred_test_1 = xgbcl_wo_spot.predict(X_test)
#Metrics for train set
eval_metrics (y_test, pred_test_1,"../../results/model/eval_metrics/", 'xgbcl_wo_spot_test')

# xgb model interpretation
plot_model_interpretation_xgb (xgbcl_wo_spot, '../../results/model/xgb_plots/', 'xgbcl_wo_spot')

xgbcl_wo_spot.save_model('../../results/model/xgb_binLog_cl_wo_spot_wo_ts.model')


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
xgb_bo = BayesianOptimization(bo_tune_xgb, { 'max_depth': (3, 5),
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

#Predicting for training set
pred_train_bo = xgbcl_bo.predict(X_train)

#Metrics for train set
eval_metrics (y_train, pred_train_bo,"../../results/model/eval_metrics/", 'xgbcl_bo_wo_spot_train')

#Predicting for test
pred_test_bo = xgbcl_bo.predict(X_test)
#Metrics for train set
eval_metrics (y_test, pred_test_bo,"../../results/model/eval_metrics/", 'xgbcl_bo_wo_spot_test')

# xgb model interpretation
plot_model_interpretation_xgb (xgbcl_bo, '../../results/model/xgb_plots/', 'xgbcl_bo_wo_spot')

xgbcl_bo.save_model('../../results/model/xgb_bo_binLog_cl_wo_spot_wo_ts.model')
