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
import shap
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


###############################
###### Cross evaluation #######
###############################
N_DATASETS = 12
# Load the 12 tuned XGBoost models (all features) and the corresponding 12 datasets
print("Loading XGBoost models and datasets...")
xgb_classifiers_tuned = [xgb.XGBClassifier(objective = 'binary:logistic') for i in range(N_DATASETS)]
[xgb_classifiers_tuned[i].load_model("models_XGB/tuned/model" + str(i) + ".json") for i in range(N_DATASETS)]

XY_xgb = []
for i in range(N_DATASETS) :
    with open('models_XGB/datasets/dataset' + str(i) + '.pkl', 'rb') as f:
        XY = pickle.load(f)
        XY_xgb.append(XY)

with open('preprocessed_datasets/data_all.pkl', 'rb') as f:
    features_df, features, targets = pickle.load(f)
print("Done.")

# Contextual row-wise and column-wise index bounds: 
models_I = len(features) # Row-wise: for the number of sex-specific splits
models_J = len(targets[0]) # Column-wise: for the number of disease-specific subset splits

# Auxiliary reshaped data structures
XY_xgb_r = np.reshape(XY_xgb, (models_I, models_J, 4))
xgb_classifiers_tuned_r = np.reshape(xgb_classifiers_tuned, (models_I, models_J))

##### Perform the cross evaluation for the tuned XGBoost models (all features) #####
print("Running cross evaluation analysis...")
fig, ax = plt.subplots(2, models_J, sharex=True, sharey=True, dpi = 160, figsize=(12, 6))
c = plt.rcParams['axes.prop_cycle'].by_key()['color'][0:2]
for j in range(models_J) :
    # Female data, both model
    x_test_xgb = XY_xgb_r[1, j][1]
    x_test_xgb['Sex'] = 0  # 0 is female
    cols = x_test_xgb.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    x_test_xgb = x_test_xgb[cols]
    y_true_xgb = XY_xgb_r[1, j][3]
    xgb_cl_tuned = xgb_classifiers_tuned_r[0, j]
    y_pred_xgb_tuned = xgb_cl_tuned.predict_proba(x_test_xgb)  # XGB test set predictions
    fpr, tpr, thresholds = roc_curve(y_true_xgb, y_pred_xgb_tuned[:, 1])  # ROC curve
    ax[0, j].plot(fpr, tpr, color=c[0])
    ax[0, j].text(0.3, 0.17, f'AUC BothXGBt = {roc_auc_score(y_true_xgb, y_pred_xgb_tuned[:, 1]):.3f}', fontsize=10,
                  color=c[0])  # AUC metric

    # Female data, female model
    x_test_xgb = XY_xgb_r[1, j][1]
    x_test_xgb.drop(columns=['Sex'], inplace=True)
    y_true_xgb = XY_xgb_r[1, j][3]
    xgb_cl_tuned = xgb_classifiers_tuned_r[1, j]
    y_pred_xgb_tuned = xgb_cl_tuned.predict_proba(x_test_xgb)  # XGB test set predictions
    fpr, tpr, thresholds = roc_curve(y_true_xgb, y_pred_xgb_tuned[:, 1])  # ROC curve
    ax[0, j].plot(fpr, tpr, color=c[1])
    ax[0, j].text(0.3, 0.1, f'AUC FemXGBt = {roc_auc_score(y_true_xgb, y_pred_xgb_tuned[:, 1]):.3f}', fontsize=10,
                  color=c[1])  # AUC metric

    # Female data, male model
    x_test_xgb = XY_xgb_r[1, j][1]
    y_true_xgb = XY_xgb_r[1, j][3]
    xgb_cl_tuned = xgb_classifiers_tuned_r[2, j]
    y_pred_xgb_tuned = xgb_cl_tuned.predict_proba(x_test_xgb)  # XGB test set predictions
    fpr, tpr, thresholds = roc_curve(y_true_xgb, y_pred_xgb_tuned[:, 1])  # ROC curve
    ax[0, j].plot(fpr, tpr, color=c[2])
    ax[0, j].text(0.3, 0.03, f'AUC MaleXGBt = {roc_auc_score(y_true_xgb, y_pred_xgb_tuned[:, 1]):.3f}', fontsize=10,
                  color=c[2])  # AUC metric

    # Male data, both model
    x_test_xgb = XY_xgb_r[2, j][1]
    x_test_xgb['Sex'] = 1  # 1 is male
    cols = x_test_xgb.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    x_test_xgb = x_test_xgb[cols]
    y_true_xgb = XY_xgb_r[2, j][3]
    xgb_cl_tuned = xgb_classifiers_tuned_r[0, j]
    y_pred_xgb_tuned = xgb_cl_tuned.predict_proba(x_test_xgb)  # XGB test set predictions
    fpr, tpr, thresholds = roc_curve(y_true_xgb, y_pred_xgb_tuned[:, 1])  # ROC curve
    ax[1, j].plot(fpr, tpr, color=c[0])
    ax[1, j].text(0.3, 0.17, f'AUC BothXGBt = {roc_auc_score(y_true_xgb, y_pred_xgb_tuned[:, 1]):.3f}', fontsize=10,
                  color=c[0])  # AUC metric

    # Male data, female model
    x_test_xgb = XY_xgb_r[2, j][1]
    x_test_xgb.drop(columns=['Sex'], inplace=True)
    y_true_xgb = XY_xgb_r[2, j][3]
    xgb_cl_tuned = xgb_classifiers_tuned_r[1, j]
    y_pred_xgb_tuned = xgb_cl_tuned.predict_proba(x_test_xgb)  # XGB test set predictions
    fpr, tpr, thresholds = roc_curve(y_true_xgb, y_pred_xgb_tuned[:, 1])  # ROC curve
    ax[1, j].plot(fpr, tpr, color=c[1])
    ax[1, j].text(0.3, 0.1, f'AUC FemXGBt = {roc_auc_score(y_true_xgb, y_pred_xgb_tuned[:, 1]):.3f}', fontsize=10,
                  color=c[1])  # AUC metric

    # Male data, male model
    x_test_xgb = XY_xgb_r[2, j][1]
    y_true_xgb = XY_xgb_r[2, j][3]
    xgb_cl_tuned = xgb_classifiers_tuned_r[2, j]
    y_pred_xgb_tuned = xgb_cl_tuned.predict_proba(x_test_xgb)  # XGB test set predictions
    fpr, tpr, thresholds = roc_curve(y_true_xgb, y_pred_xgb_tuned[:, 1])  # ROC curve
    ax[1, j].plot(fpr, tpr, color=c[2])
    ax[1, j].text(0.3, 0.03, f'AUC MaleXGBt = {roc_auc_score(y_true_xgb, y_pred_xgb_tuned[:, 1]):.3f}', fontsize=10,
                  color=c[2])  # AUC metric

    # Formatting
    ax[0, j].plot(np.linspace(0, 1, 100), np.linspace(0,1,100),'--k')
    ax[1, j].plot(np.linspace(0, 1, 100), np.linspace(0,1,100),'--k')
    ax[0, j].set_xlabel('Female Data FPR', fontsize = 12)
    ax[1, j].set_xlabel('Male Data FPR', fontsize = 12)
    ax[0, j].set_ylabel(('Female Data TPR' if j == 0 else ''), fontsize = 12)
    ax[1, j].set_ylabel(('Male Data TPR' if j == 0 else ''), fontsize = 12)
    ax[0, j].set(ylim = [0., 1.])
    ax[1, j].set(ylim = [0., 1.])
    ax[0, j].set_aspect('equal', 'box')
    ax[1, j].set_aspect('equal', 'box')

fig.tight_layout()
print("Done.")

print("Saving figure...")
fig.savefig("figures/Cross_Evaluation/cross_evaluation.png", dpi = 600)
print("Done.")