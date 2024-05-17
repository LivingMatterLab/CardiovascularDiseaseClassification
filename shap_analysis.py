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


###########################
###### XGBoost SHAP #######
###########################
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
print("Done.")

##### Perform SHAP analysis for 12 datasets and save the figures #####
print("Running SHAP analysis...")
# Setup and run the SHAP feature explainer for the tuned XGBoost model [0, 0] (BA variant)
explainer_xgbu_tuned = shap.TreeExplainer(xgb_classifiers_tuned[0])
shap_values_xgbu_tuned = explainer_xgbu_tuned.shap_values(XY_xgb[0][1])
shap.summary_plot(shap_values_xgbu_tuned, XY_xgb[0][1], max_display = 10, show = False, plot_size = (11.0, 6.0))
plt.savefig('figures/SHAP/BA_tuned.png', dpi = 600)
plt.clf()
# Both, Hyp
explainer_xgbu_tuned = shap.TreeExplainer(xgb_classifiers_tuned[1])
shap_values_xgbu_tuned = explainer_xgbu_tuned.shap_values(XY_xgb[1][1])
shap.summary_plot(shap_values_xgbu_tuned, XY_xgb[1][1], max_display = 10, show = False)
plt.savefig('figures/SHAP/BH_tuned.png', dpi = 600)
plt.clf()
# Both, Isch
explainer_xgbu_tuned = shap.TreeExplainer(xgb_classifiers_tuned[2])
shap_values_xgbu_tuned = explainer_xgbu_tuned.shap_values(XY_xgb[2][1])
shap.summary_plot(shap_values_xgbu_tuned, XY_xgb[2][1], max_display = 10, show = False)
plt.savefig('figures/SHAP/BI_tuned.png', dpi = 600)
plt.clf()
# Both, Con
explainer_xgbu_tuned = shap.TreeExplainer(xgb_classifiers_tuned[3])
shap_values_xgbu_tuned = explainer_xgbu_tuned.shap_values(XY_xgb[3][1])
shap.summary_plot(shap_values_xgbu_tuned, XY_xgb[3][1], max_display = 10, show = False)
plt.savefig('figures/SHAP/BC_tuned.png', dpi = 600)
plt.clf()
# Fem, Any
explainer_xgbu_tuned = shap.TreeExplainer(xgb_classifiers_tuned[4])
shap_values_xgbu_tuned = explainer_xgbu_tuned.shap_values(XY_xgb[4][1])
shap.summary_plot(shap_values_xgbu_tuned, XY_xgb[4][1], max_display = 10, show = False)
plt.savefig('figures/SHAP/FA_tuned.png', dpi = 600)
plt.clf()
# Fem, Hyp
explainer_xgbu_tuned = shap.TreeExplainer(xgb_classifiers_tuned[5])
shap_values_xgbu_tuned = explainer_xgbu_tuned.shap_values(XY_xgb[5][1])
shap.summary_plot(shap_values_xgbu_tuned, XY_xgb[5][1], max_display = 10, show = False)
plt.savefig('figures/SHAP/FH_tuned.png', dpi = 600)
plt.clf()
# Fem, Isch
explainer_xgbu_tuned = shap.TreeExplainer(xgb_classifiers_tuned[6])
shap_values_xgbu_tuned = explainer_xgbu_tuned.shap_values(XY_xgb[6][1])
shap.summary_plot(shap_values_xgbu_tuned, XY_xgb[6][1], max_display = 10, show = False)
plt.savefig('figures/SHAP/FI_tuned.png', dpi = 600)
plt.clf()
# Fem, Con
explainer_xgbu_tuned = shap.TreeExplainer(xgb_classifiers_tuned[7])
shap_values_xgbu_tuned = explainer_xgbu_tuned.shap_values(XY_xgb[7][1])
shap.summary_plot(shap_values_xgbu_tuned, XY_xgb[7][1], max_display = 10, show = False)
plt.savefig('figures/SHAP/FC_tuned.png', dpi = 600)
plt.clf()
# Male, Any
explainer_xgbu_tuned = shap.TreeExplainer(xgb_classifiers_tuned[8])
shap_values_xgbu_tuned = explainer_xgbu_tuned.shap_values(XY_xgb[8][1])
shap.summary_plot(shap_values_xgbu_tuned, XY_xgb[8][1], max_display = 10, show = False)
plt.savefig('figures/SHAP/MA_tuned.png', dpi = 600)
plt.clf()
# Male, Hyp
explainer_xgbu_tuned = shap.TreeExplainer(xgb_classifiers_tuned[9])
shap_values_xgbu_tuned = explainer_xgbu_tuned.shap_values(XY_xgb[9][1])
shap.summary_plot(shap_values_xgbu_tuned, XY_xgb[9][1], max_display = 10, show = False)
plt.savefig('figures/SHAP/MH_tuned.png', dpi = 600)
plt.clf()
# Male, Isch
explainer_xgbu_tuned = shap.TreeExplainer(xgb_classifiers_tuned[10])
shap_values_xgbu_tuned = explainer_xgbu_tuned.shap_values(XY_xgb[10][1])
shap.summary_plot(shap_values_xgbu_tuned, XY_xgb[10][1], max_display = 10, show = False)
plt.savefig('figures/SHAP/MI_tuned.png', dpi = 600)
plt.clf()
# Male, Con
explainer_xgbu_tuned = shap.TreeExplainer(xgb_classifiers_tuned[11])
shap_values_xgbu_tuned = explainer_xgbu_tuned.shap_values(XY_xgb[11][1])
shap.summary_plot(shap_values_xgbu_tuned, XY_xgb[11][1], max_display = 10, show = False)
plt.savefig('figures/SHAP/MC_tuned.png', dpi = 600)
plt.clf()
print("Done.")

