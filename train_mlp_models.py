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
######################################


#########################
###### MLP models #######
#########################
# Load the relevant datasets
print("Loading datasets...")
with open('preprocessed_datasets/data_all.pkl', 'rb') as f:
    features_df, features, targets = pickle.load(f)
print("Done.")

# Build 12 MLP models
TEST_SIZE = 0.15 # Test size defined here for consistency later in SHAP; note: val size = test size
MAX_EPOCHS = 100 # 100
print("Initializing MLP models...")
models_mlp = [build_baseline_MLP(features[i], Y, valtest_frac = TEST_SIZE * 2, max_epochs = MAX_EPOCHS, oversample = True) for i in range(len(features)) for Y in targets[i]]

# models_mlp = [build_baseline_MLP(features[0], targets[0][0], valtest_frac = TEST_SIZE * 2, max_epochs = MAX_EPOCHS, oversample = True)]
print("Done.")

# Train the 12 MLP models
print("Training MLP models...")
history = [dnn[0].fit(dnn[1], steps_per_epoch = dnn[4], epochs = dnn[5], validation_data = dnn[2], verbose = 1) for dnn in models_mlp]
print("Done.")

# Save the 12 MLP models
print("Saving MLP models...")
i = 0
for model in models_mlp :
    model[0].save('models_MLP/mlp' + str(i))
    tf.data.Dataset.save(model[3], 'models_MLP/test_ds' + str(i))
    i = i + 1
print("Done.")