# #### Resources used:
# - **UK Biobank (data)**: https://www.ukbiobank.ac.uk/
# - **Tensorflow**: https://www.tensorflow.org/tutorials
# - **Keras**: https://keras.io/examples/
# - **XGBoost**: repo: https://github.com/dmlc/xgboost, doc: https://xgboost.readthedocs.io/en/stable/tutorials/index.html
# - **SHAP**: https://shap.readthedocs.io/en/latest/tabular_examples.html
# - **joblib Parallel**: https://joblib.readthedocs.io/en/latest/generated/joblib.Parallel.html
# - **scikit-learn**: https://scikit-learn.org/stable/modules/classes.html


#######################
###### Packages #######
#######################
import numpy as np
import pandas as pd
import tensorflow as tf
import seaborn as sns
import xgboost as xgb
from joblib import Parallel, delayed
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV

import openml
from openml.datasets.functions import create_dataset
from matplotlib import pyplot as plt
######################################

##############################
########## XGBoost ###########
##############################
# Helper function for parallelized XGBoost training
def fit_xgb(x_train, y_train) :
    xgb_cl = xgb.XGBClassifier(objective = "binary:logistic")
    xgb_cl.fit(x_train, y_train)
    
    return xgb_cl

# Tunes the XGBoost model on a given dataset (x_train, y_train) using k-fold cross-validation
# and optimizing for the AUC score
def tune_xgb(x_train, y_train, params, n_iter = 25) :
    xgb_cl = xgb.XGBClassifier(objective = "binary:logistic")
    rand_cv = RandomizedSearchCV(xgb_cl, params, n_iter = 25, scoring="roc_auc", verbose = 1)
    rand_cv.fit(x_train, y_train)
    
    # get best model from randomized search
    xgb_tuned = rand_cv.best_estimator_
    return xgb_tuned