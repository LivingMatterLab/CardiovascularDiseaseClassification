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


###################################
###### Build SAINT Datasets #######
###################################
# Load the relevant datasets
print("Loading datasets...")
# All features
with open('preprocessed_datasets/data_all.pkl', 'rb') as f:
    features_df, features, targets = pickle.load(f)
print("Done.")

# Split the SAINT dataset into train-dev-test with the same test set as MLP and XGBoost
print("Building SAINT datasets...")
TEST_SIZE = 0.15
XY_saint = [train_test_split(features_df[i], Y, test_size = TEST_SIZE, shuffle = False) for i in range(len(features_df)) for Y in targets[i]]
for i in range(len(XY_saint)) :
    temp = train_test_split(XY_saint[i][0], XY_saint[i][2], test_size = TEST_SIZE / (1 - TEST_SIZE), shuffle = False) # Split 50-50 for dev and test sets
    XY_saint[i] = [temp[0], temp[1], XY_saint[i][1], temp[2], temp[3], XY_saint[i][3]]
    # (x_train, x_val, x_test, y_train, y_val, y_test)

# Positive class oversampling for all SAINT model variants
for i in range(len(XY_saint)) :
    XY_saint[i][0], XY_saint[i][3] = oversample_train(XY_saint[i][0], XY_saint[i][3])

# Save the processed SAINT datasets
i = 0
for XY in XY_saint :
    save_dataset_saint('SAINT/saint_datasets/ds_saint' + str(i) + '.pkl', XY)
    i = i + 1
print("Done.")
