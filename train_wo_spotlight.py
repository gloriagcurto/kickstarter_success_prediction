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
import shap
from sklearn.metrics import classification_report, confusion_matrix
from bayes_opt import BayesianOptimization



def plot_model_interpretation_xgb (model, output_path, file_prefix):
    '''
    xgboost model interpretation plots 
    '''
    output_filename = output_path + file_prefix + "_tree.png"
    xgb.plot_tree(model, num_trees=2)
    fig = plt.gcf()
    fig.set_size_inches(15, 10)
    plt.title("Model decision tree")
    fig.savefig(output_filename)
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
    report = classification_report(y_true, y_pred , output_dict=True)
    print(report)
    pd.DataFrame(report).transpose().to_latex(buf=output_filename)

    #confusion matrix
    cm= confusion_matrix(y_true, y_pred)
    print(f'Confusion matrix: {cm}')
    output_filename = output_path + file_prefix + "_confusion_matrix.tex"
    pd.DataFrame(cm).to_latex(buf=output_filename)


def bo_tune_xgb(max_depth, gamma, subsample, learning_rate):
    '''
    Bayesian Optimization function for xgboost
    Specify the parameters you want to tune as keyword arguments
    '''
    params = {'objective': 'binary:logistic',
              'max_depth': int(max_depth),
              'gamma': gamma,
              'learning_rate':learning_rate,
              'subsample': subsample,
              'eval_metric': 'auc'}

    #Cross validating with the specified parameters in 5 folds and 100 iterations
    cv_results = xgb.cv(params, dtrain,  num_boost_round=100, nfold=5, early_stopping_rounds=100, as_pandas=True,  seed=37)
    return cv_results["test-auc-mean"].iloc[-1]


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

# Hyper parameter bayesian optimization 
#Invoking the Bayesian Optimizer with the specified parameters to tune
xgb_bo = BayesianOptimization(bo_tune_xgb, { 'max_depth': (3, 8), # default 6
                                             'gamma': (0, 5), # default 0
                                             'learning_rate':(0, 1), #default 0.3
                                             'subsample':(0, 1) # default 1
                                            })
#print(xgb_bo.columns)

#performing Bayesian optimization for 5 iterations with 8 steps of random exploration with an #acquisition function of expected improvement
xgb_bo.maximize(n_iter=5, init_points=8, acq='ei')

#Extracting the best parameters
params = xgb_bo.max['params']
print(params)

#Converting the max_depth and n_estimator values from float to int
params['max_depth']= int(params['max_depth'])

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


'''
Default values model works a bit better
'''
# model interpretation with SHAP
e_default = shap.TreeExplainer(xgbcl_wo_spot)
shap_values = e_default.shap_values(X_test)
#shap_interaction_values = e_default.shap_interaction_values(X_test)
#X_display = X_train.columns

#Bar chart of mean importance
fig = plt.figure()
shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
plt.savefig('../../results/model/shap_plots/bar_chart_mean_importance_xgbcl_wo_spot_test.pdf', bbox_inches='tight')
plt.close(fig)

#Summary plot
fig = plt.figure()
shap.summary_plot(shap_values, X_test, plot_type='dot', show=False)
plt.savefig('../../results/model/shap_plots/summary_plot_xgbcl_wo_spot_test.pdf', bbox_inches='tight')
plt.close(fig)


'''
#Dependence plot
for name in X_train.columns:
    output_filename= '../../results/model/shap_plots/dependence_' + name + '_xgbcl_wo_spot_test.pdf'
    shap.dependence_plot(name, shap_values, X_test)
    plt.savefig(output_filename)
    plt.close(fig)
'''
#force_plot
fig = plt.figure()
shap.force_plot(e_default.expected_value, shap_values[0,:], X_test.iloc[0,:],show=False,matplotlib=True)
plt.savefig('../../results/model/shap_plots/force_plot_xgbcl_wo_spot_test.pdf', bbox_inches='tight')
plt.close(fig)
#shap.force_plot(e_default.expected_value, shap_values[0:500,:], X_test[0:500,:])

'''
shap.waterfall_plot(*args, **kwargs)
shap.image_plot(*args, **kwargs)
shap.decision_plot(*args, **kwargs)
'''