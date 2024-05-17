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
# import keras_tuner as kt
import xgboost as xgb
# import shap
from joblib import Parallel, delayed
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV

import openml
from openml.datasets.functions import create_dataset

from matplotlib import pyplot as plt
import pickle

from aux_functions_data import preprocess_df, buildXY, oversample_train, save_dataset_saint
from aux_functions_mlp import build_baseline_MLP
from aux_functions_xgb import fit_xgb, tune_xgb
######################################


#############################
###### XGBoost models #######
#############################
# Load the relevant datasets
print("Loading datasets...")
# All features
with open('preprocessed_datasets/data_all.pkl', 'rb') as f:
    features_df, features, targets = pickle.load(f)
# Framingham only
with open('preprocessed_datasets/data_F.pkl', 'rb') as f:
    features_dfF, featuresF, targetsF = pickle.load(f)
print("Done.")

# Train-test split for XGBoost
TEST_SIZE = 0.15 # Changed to 0.2 for consistency
print("Initializing XGBoost datasets...")
XY_xgb = [train_test_split(features_df[i], Y, test_size = TEST_SIZE, shuffle = False) for i in range(len(features_df)) for Y in targets[i]]
XY_xgbF = [train_test_split(features_dfF[i], Y, test_size = TEST_SIZE, shuffle = False) for i in range(len(features_dfF)) for Y in targetsF[i]]
# Each element of XY_xgb: (x_train, x_test, y_train, y_test)

# Positive class oversampling for all XGBoost model variants
for i in range(len(XY_xgb)) :
    XY_xgb[i][0], XY_xgb[i][2] = oversample_train(XY_xgb[i][0], XY_xgb[i][2])
for i in range(len(XY_xgbF)) :
    XY_xgbF[i][0], XY_xgbF[i][2] = oversample_train(XY_xgbF[i][0], XY_xgbF[i][2])
print("Done.")

# Save XGBoost datasets
print("Saving all XGBoost datasets...")
for i in range(len(XY_xgb)):
    with open('models_XGB/datasets/dataset' + str(i) + '.pkl', 'wb') as f:
        pickle.dump(XY_xgb[i], f)
    with open('models_XGB_Fram/datasets/dataset' + str(i) + '.pkl', 'wb') as f:
        pickle.dump(XY_xgbF[i], f)
print("Done.")

#### Train 12 XGBoost models (untuned) for both all features and Framingham features ####
print("Training 12 XGBoost model variants (untuned) for all features...")
xgb_classifiers = Parallel(n_jobs = 16)(delayed(fit_xgb)(XY_xgb[i][0], XY_xgb[i][2]) for i in range(len(XY_xgb)))
print("Done.")
print("Training 12 XGBoost model variants (untuned) for Framingham only...")
xgb_classifiersF = Parallel(n_jobs = 16)(delayed(fit_xgb)(XY_xgbF[i][0], XY_xgbF[i][2]) for i in range(len(XY_xgbF)))
print("Done.")

######### XGBoost tuning #########
print("Starting XGBoost tuning...")
# XGBoost hyperparameter choices for the randomized search
params = {'max_depth': [1, 3, 5, 6, 10, 15, 20],
          'learning_rate': [0.001, 0.01, 0.1, 0.2, 0.3],
          'subsample': np.arange(0.3, 1.0, 0.1),
          'colsample_bytree': np.arange(0.4, 1.0, 0.1),
          'colsample_bylevel': np.arange(0.4, 1.0, 0.1),
          'n_estimators': [100, 500, 1000, 1500]}

# Perform the XGBoost tuning for all 12 dataset variants; Set n_jobs to the number of desired processes
print("Tuning the 12 XGBoost models for all features...")
xgb_classifiers_tuned = Parallel(n_jobs = 12)(delayed(tune_xgb)(XY_xgb[i][0], XY_xgb[i][2], params) for i in range(len(XY_xgb)))
print("Done.")

# Perform the XGBoost tuning on Framingham Only for all 12 dataset variants; Set n_jobs to the number of desired processes
print("Tuning the 12 XGBoost models for Framingham only...")
xgb_classifiers_tunedF = Parallel(n_jobs = 12)(delayed(tune_xgb)(XY_xgbF[i][0], XY_xgbF[i][2], params) for i in range(len(XY_xgbF)))
print("Done.")
print("Done tuning.")

# Save XGBoost models to json
print("Saving all XGBoost models...")
for i in range(len(XY_xgb)):
    xgb_classifiers[i].save_model("models_XGB/untuned/model" + str(i) + ".json")
    xgb_classifiersF[i].save_model("models_XGB_Fram/untuned/model" + str(i) + ".json")

    xgb_classifiers_tuned[i].save_model("models_XGB/tuned/model" + str(i) + ".json")
    xgb_classifiers_tunedF[i].save_model("models_XGB_Fram/tuned/model" + str(i) + ".json")
print("Done.")

