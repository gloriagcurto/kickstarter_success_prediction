#!/usr/bin/python
# -*- coding:utf-8 -*-

"""
SHAP model explanation Kickstarter_2020-02-13T03_20_04_893Z.
Decision plos and evaluation of wrong predictions
Input Filename: train, test , train_ts, test_ts
Output Filename: 
https://github.com/slundberg/shap/blob/master/notebooks/plots/decision_plot.ipynb
Copyright (c) 2020 Gloria G. Curto.
http://gloriagcurto.info
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix
import shap
import warnings
from joblib import dump, load

# Read train and test files and drop the variable spotlight due to its high correlation to the target

X_train = pd.read_hdf('../../data/data_model/X_train.h5')
X_train.drop( 'spotlight', axis=1, inplace=True)
y_train = pd.read_hdf('../../data/data_model/y_train.h5')

X_test =  pd.read_hdf('../../data/data_model/X_test.h5')
X_test.drop( 'spotlight', axis=1, inplace=True)
y_test =  pd.read_hdf('../../data/data_model/y_test.h5')

dtrain = xgb.DMatrix(data=X_train, label=y_train)
dtest = xgb.DMatrix(data=X_test, label=y_test)

#load trained model
#xgbcl_wo_spot = xgb.XGBClassifier.load_model(fname='../../results/model/xgb_binLog_cl_wo_spot_wo_ts.model')

xgbcl_wo_spot = load('../../results/model/xgb_binLog_cl_wo_spot_wo_ts.joblib') 


y_pred = xgbcl_wo_spot.predict(X_test)

explainer = shap.TreeExplainer(xgbcl_wo_spot)

expected_value = explainer.expected_value
if isinstance(expected_value, list):
    expected_value = expected_value[1]
#print(f"Explainer expected value: {expected_value}")

#select = range(100)
features = X_test   #.iloc[select]
features_display = X_test.columns

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    shap_values = explainer.shap_values(features)[1]
    shap_interaction_values = explainer.shap_interaction_values(features)
if isinstance(shap_interaction_values, list):
    shap_interaction_values = shap_interaction_values[1]
'''
#basic 
output_filename= '../../results/model/shap_plots/decision/decision_basic_xgbcl_wo_spot_test.pdf'
shap.decision_plot(expected_value, shap_values, features_display, show=False)
fig = plt.gcf()
fig.set_size_inches(15, 10)
plt.title(f"Decision plot")
plt.tight_layout()
fig.savefig(output_filename)
plt.close(fig)

#probabilities
output_filename= '../../results/model/shap_plots/decision/decision_logit_xgbcl_wo_spot_test.pdf'
shap.decision_plot(expected_value, shap_values, features_display, link='logit', show=False)
fig = plt.gcf()
fig.set_size_inches(15, 10)
plt.title(f"Decision plot")
plt.tight_layout()
fig.savefig(output_filename)
plt.close(fig)
'''

well_classified = (y_pred == y_test)
classif= pd.concat([well_classified, pd.Series(y_pred, index=y_test.index), y_test], axis=1)
classif.columns = ['well', 'y_pred', 'y_test']
misclassified= classif.loc[classif['well']==False]

print(len(misclassified))
print(misclassified.head())

'''
output_filename= '../../results/model/shap_plots/decision/decision_logit_miscla_xgbcl_wo_spot_test.pdf'
shap.decision_plot(expected_value, shap_values, features_display, link='logit', highlight=misclassified, show=False)
fig = plt.gcf()
fig.set_size_inches(15, 10)
plt.title(f"Decision plot  of misclassified observation")
plt.tight_layout()
fig.savefig(output_filename)
plt.close(fig)

#plot individual missclassifications

output_filename= '../../results/model/shap_plots/decision/decision_logit_miscla_ind_xgbcl_wo_spot_test.pdf'
shap.decision_plot(expected_value, shap_values[misclassified], features_display[misclassified],
                   link='logit', highlight=0, show=False)
fig = plt.gcf()
fig.set_size_inches(15, 10)
plt.title(f"Decision plot  of misclassified observations")
plt.tight_layout()
fig.savefig(output_filename)
plt.close(fig)
'''

#force plot misclassffications  suc as failed
output_filename= '../../results/model/shap_plots/force/force_logit_miscla_xgbcl_wo_spot_test_10146_suc_pred_fail.pdf'
shap.force_plot(expected_value, shap_values, X_test.loc[10146,:], 
                link='logit', show=False, matplotlib=True)
fig = plt.gcf()
fig.set_size_inches(15, 10)
plt.title(f"Force plot of misclassified observation 10146:\nsuccessful project predicted as failed")
plt.tight_layout()
fig.savefig(output_filename)
plt.close(fig)  

#force plot misclassffications  failed as suc
output_filename= '../../results/model/shap_plots/force/force_logit_miscla_xgbcl_wo_spot_test_121366_fail_pred_suc.pdf'
shap.force_plot(expected_value, shap_values, X_test.loc[121366,:], 
                link='logit', show=False, matplotlib=True)
fig = plt.gcf()
fig.set_size_inches(15, 10)
plt.title(f"Force plot of misclassified observation 121366:\nfailed project predicted as successful")
plt.tight_layout()
fig.savefig(output_filename)
plt.close(fig)  
'''
# general
output_filename= '../../results/model/shap_plots/force/force_logit_miscla_xgbcl_wo_spot_test.pdf'
shap.force_plot(expected_value, shap_values, features_display[misclassified], 
                link='logit', show=False, matplotlib=True)

fig = plt.gcf()
fig.set_size_inches(15, 10)
plt.title(f"Force plot of misclassified observations")
plt.tight_layout()
fig.savefig(output_filename)
plt.close(fig)   

#Identify typical prediction paths
#A decision plot can expose a model's typical prediction paths. Here, we plot all of the predictions in the probability interval [0.98, 1.0]
# to see what high-scoring predictions have in common. We use 'hclust' feature ordering to group similar prediction paths. The plot shows
# two distinct paths. The effects of  are also notable.

# Get predictions on the probability scale.
T = X_test[y_pred >= 0.98]
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    sh = explainer.shap_values(T)[1]

output_filename= '../../results/model/shap_plots/decision/typical_decision_path_logit_xgbcl_wo_spot_test_proba0_98.pdf'
shap.decision_plot(expected_value, sh, T, feature_order='hclust', link='logit', show=False)
fig = plt.gcf()
fig.set_size_inches(15, 10)
plt.title(f"Typical decision path for successful projects")
plt.tight_layout()
fig.savefig(output_filename)
plt.close(fig) 


T = X_test[y_pred <= 0.2]
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    sh = explainer.shap_values(T)[1]

output_filename= '../../results/model/shap_plots/decision/typical_decision_path_logit_xgbcl_wo_spot_test_proba0_2.pdf'
shap.decision_plot(expected_value, sh, T, feature_order='hclust', link='logit', show=False, matplotlib=True)
fig = plt.gcf()
fig.set_size_inches(15, 10)
plt.title(f"Typical decision path for failed projects")
plt.tight_layout()
fig.savefig(output_filename)
plt.close(fig) 
'''