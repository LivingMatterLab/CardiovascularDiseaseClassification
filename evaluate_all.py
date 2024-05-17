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
import torch
from joblib import Parallel, delayed
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV

import openml
from openml.datasets.functions import create_dataset

from matplotlib import pyplot as plt
import pickle

import sys
sys.path.insert(0, './SAINT')
from models import SAINT
from augmentations import embed_data_mask

from aux_functions_data import preprocess_df, buildXY, oversample_train, save_dataset_saint
from aux_functions_mlp import build_baseline_MLP
######################################

####################################
###### Data and model import #######
####################################
N_DATASETS = 12

# Load the relevant datasets
print("Loading preprocessed datasets...")
with open('preprocessed_datasets/data_all.pkl', 'rb') as f:
    features_df, features, targets = pickle.load(f)
print("Done.")

# Contextual row-wise and column-wise index bounds: 
models_I = len(features) # Row-wise: for the number of sex-specific splits
models_J = len(targets[0]) # Column-wise: for the number of disease-specific subset splits

# Load the 12 MLP models
print("Loading MLP models...")
models_mlp = []
for i in range(N_DATASETS) :
    models_mlp.append([tf.keras.models.load_model('models_MLP/mlp' + str(i)), 
                       tf.data.Dataset.load('models_MLP/test_ds' + str(i))])
print("Done.")

# Load the 36 XGBoost models (24 (12 untuned + 12 tuned) for all features, 12 tuned for Framingham only)
# and the corresponding 24 (12 all features + 12 Framingham) datasets
print("Loading XGBoost models and datasets...")
xgb_classifiers = [xgb.XGBClassifier(objective = 'binary:logistic') for i in range(N_DATASETS)]
[xgb_classifiers[i].load_model("models_XGB/untuned/model" + str(i) + ".json") for i in range(N_DATASETS)]
xgb_classifiers_tunedF = [xgb.XGBClassifier(objective = 'binary:logistic') for i in range(N_DATASETS)]
[xgb_classifiers_tunedF[i].load_model("models_XGB_Fram/tuned/model" + str(i) + ".json") for i in range(N_DATASETS)]
xgb_classifiers_tuned = [xgb.XGBClassifier(objective = 'binary:logistic') for i in range(N_DATASETS)]
[xgb_classifiers_tuned[i].load_model("models_XGB/tuned/model" + str(i) + ".json") for i in range(N_DATASETS)]

XY_xgb = []
XY_xgbF = []
for i in range(N_DATASETS) :
    with open('models_XGB/datasets/dataset' + str(i) + '.pkl', 'rb') as f:
        XY = pickle.load(f)
        XY_xgb.append(XY)
    with open('models_XGB_Fram/datasets/dataset' + str(i) + '.pkl', 'rb') as f:
        XYF = pickle.load(f)
        XY_xgbF.append(XYF)
print("Done.")

# Load the 12 SAINT models and the corresponding 12 datasets
print("Loading SAINT models and datasets...")
models_saint = []
testloaders = []
vision_dsets = []
XY_saint = []
for i in range(N_DATASETS) :
    with open('bestmodels_saint/binary/ds_saint' + str(i) + '.pkl/testrun/properties.pkl', 'rb') as f:
        p = pickle.load(f)
    model = SAINT(categories = p[0][0], 
                    num_continuous = p[0][1],                
                    dim = p[0][2],                           
                    dim_out = p[0][3],                       
                    depth = p[0][4],                       
                    heads = p[0][5],                         
                    attn_dropout = p[0][6],             
                    ff_dropout = p[0][7],                  
                    mlp_hidden_mults = p[0][8],       
                    cont_embeddings = p[0][9],
                    attentiontype = p[0][10],
                    final_mlp_style = p[0][11],
                    y_dim = p[0][12])
    testloaders.append(p[1][2])
    vision_dsets.append(p[2])
    model.load_state_dict(torch.load('bestmodels_saint/binary/ds_saint' + str(i) + '.pkl/testrun/bestmodel.pth'))
    models_saint.append(model)
    # models_saint = [torch.load('bestmodels_saint/binary/ds_saint' + str(i) + '.pkl/testrun/bestmodel.pth') for i in range(1, 2)]
    with open('SAINT/saint_datasets/train_val_test' + str(i) + '.pkl', 'rb') as f:
        XY = pickle.load(f)
        XY_saint.append(XY)
print("Done.")

##################################
###### Evaluate all models #######
##################################
################
## MLP models ##
################
print("Evaluating MLP models...")
# Make predictions on the corresponding TEST SET
print("Computing predictions and extracting ground truth on the test set...")
y_pred = [mlp[0].predict(mlp[1], verbose=0) for mlp in models_mlp]
y_pred = np.reshape(y_pred, (models_I, models_J)) # Reshape collection to 2D model grid, as before
print("Done.")

# Extract the TEST SET ground truth
XY_test = [tuple(zip(*mlp[1])) for mlp in models_mlp]
y_true = [np.array(xy_test[1]) for xy_test in XY_test]
y_true = np.reshape(y_true, (models_I, models_J))

# Plot the ROC curve and report the AUC metric for all 12 MLPs
print("Plotting the ROC curves...")
fig, ax = plt.subplots(models_I, models_J, sharex=True, sharey=True, dpi = 160, figsize=(12, 8.5))

for i in range(models_I) :
    for j in range(models_J) :
        fpr, tpr, thresholds = roc_curve(y_true[i, j], y_pred[i, j]) # ROC curve
        ax[i, j].plot(fpr, tpr)
        
        # Formatting
        ax[i, j].plot(np.linspace(0,1, 100), np.linspace(0,1,100),'--r')
        ax[i, j].set_xlabel(('FPR' if i == 2 else ''), fontsize = 12)
        ax[i, j].set_ylabel(('TPR' if j == 0 else ''), fontsize = 12)
        ax[i, j].set(ylim = [0., 1.])
        ax[i, j].set_aspect('equal', 'box')
        ax[i, j].text(0.5, 0.2, f'AUC = {roc_auc_score(y_true[i, j], y_pred[i, j]):.3f}', fontsize = 12) # AUC metric

fig.tight_layout()
print("Done.")

# Save fig.
print("Saving figure...")
fig.savefig("figures/ROC_MLP.png", dpi = 600) 
print("Done.")
print("Done evaluating MLP.")

####################
## XGBoost models ##
####################
## Untuned ##
print("Evaluating XGBoost models (untuned)...")

# Reshape the comprehensive XGBoost model collection 
# and the (untuned) trained XGBoost classifier collection as 3x4 grids
# (rows = sex-specific splits, columns = disease specific splits) 
XY_xgb_r = np.reshape(XY_xgb, (models_I, models_J, 4))
xgb_classifiers_r = np.reshape(xgb_classifiers, (models_I, models_J))

# Plot the ROC curve and report the AUC metric for all 12 untuned XGBoost models
print("Plotting the ROC curves...")
fig, ax = plt.subplots(models_I, models_J, sharex=True, sharey=True, dpi = 160, figsize=(12, 8.5))
for i in range(models_I) :
    for j in range(models_J) :
        # Make XGBoost predictions on the test set
        xgb_cl = xgb_classifiers_r[i, j]
        x_test_xgb = XY_xgb_r[i, j][1]
        y_true_xgb = XY_xgb_r[i, j][3]
        y_pred_xgb = xgb_cl.predict_proba(x_test_xgb)
        
        # ROC curve
        fpr, tpr, thresholds = roc_curve(y_true_xgb, y_pred_xgb[:,1])
        ax[i, j].plot(fpr, tpr)
        
        # Formatting
        ax[i, j].plot(np.linspace(0,1, 100), np.linspace(0,1,100),'--r')
        ax[i, j].set_xlabel(('FPR' if i == 2 else ''), fontsize = 12)
        ax[i, j].set_ylabel(('TPR' if j == 0 else ''), fontsize = 12)
        ax[i, j].set(ylim = [0., 1.])
        ax[i, j].set_aspect('equal', 'box')
        ax[i, j].text(0.5, 0.2, f'AUC = {roc_auc_score(y_true_xgb, y_pred_xgb[:,1]):.3f}', fontsize = 12) # AUC metric
        
fig.tight_layout()
print("Done.")

# Save fig.
print("Saving figure...")
fig.savefig("figures/ROC_XGB_Untuned.png", dpi = 600)
print("Done.")
print("Done evaluating XGBoost (untuned).")

## Tuned ##
print("Evaluating XGBoost models (tuned)...")

# Reshape the comprehensive XGBoost model collection 
# and the (tuned) trained XGBoost classifier collection as 3x4 grids
# (rows = sex-specific splits, columns = disease specific splits) 
XY_xgb_r = np.reshape(XY_xgb, (models_I, models_J, 4))
XY_xgb_rF = np.reshape(XY_xgbF, (models_I, models_J, 4))
xgb_classifiers_tuned_r = np.reshape(xgb_classifiers_tuned, (models_I, models_J))
xgb_classifiers_tuned_rF = np.reshape(xgb_classifiers_tunedF, (models_I, models_J))

# Plot the ROC curve and report the AUC metric for all 24 individually tuned XGBoost models 
# (12 for all features, and 12 for Framingham)
print("Plotting the ROC curves...")
fig, ax = plt.subplots(models_I, models_J, sharex=True, sharey=True, dpi = 160, figsize=(12, 8.5))
for i in range(models_I) :
    for j in range(models_J) :
        # Make XGBoost predictions on the test set
        x_test_xgb = XY_xgb_r[i, j][1]
        y_true_xgb = XY_xgb_r[i, j][3]
        xgb_cl_tuned = xgb_classifiers_tuned_r[i, j]
        y_pred_xgb_tuned = xgb_cl_tuned.predict_proba(x_test_xgb)
        # framingham only
        x_test_xgbF = XY_xgb_rF[i, j][1]
        y_true_xgbF = XY_xgb_rF[i, j][3]
        xgb_cl_tunedF = xgb_classifiers_tuned_rF[i, j]
        y_pred_xgb_tunedF = xgb_cl_tunedF.predict_proba(x_test_xgbF)
        
        # ROC curve
        fpr, tpr, thresholds = roc_curve(y_true_xgb, y_pred_xgb_tuned[:,1])
        ax[i, j].plot(fpr, tpr)
        fprF, tprF, thresholds = roc_curve(y_true_xgbF, y_pred_xgb_tunedF[:,1])
        ax[i, j].plot(fprF, tprF)
        
        # Formatting
        ax[i, j].plot(np.linspace(0, 1, 100), np.linspace(0,1,100),'--r')
        ax[i, j].set_xlabel(('FPR' if i == 2 else ''), fontsize = 12)
        ax[i, j].set_ylabel(('TPR' if j == 0 else ''), fontsize = 12)
        ax[i, j].set(ylim = [0., 1.])
        ax[i, j].set_aspect('equal', 'box')
        ax[i, j].text(0.3, 0.2, f'AUC = {roc_auc_score(y_true_xgb, y_pred_xgb_tuned[:,1]):.3f}', fontsize = 12)
        ax[i, j].text(0.3, 0.1, f'AUC Fram = {roc_auc_score(y_true_xgbF, y_pred_xgb_tunedF[:,1]):.3f}', fontsize = 12) # AUC metric
        
fig.tight_layout()
print("Done.")

# Save fig.
print("Saving figure...")
fig.savefig("figures/ROC_XGB_Tuned.png", dpi = 600)
print("Done.")
print("Done evaluating XGBoost (tuned).")

##################
## SAINT models ##
##################
print("Evaluating SAINT models...")
device = torch.device("cpu")

print("Computing predictions and extracting ground truth on the test set...")
y_test_saint = [[0 for x in range(models_J)] for x in range(models_I)]
y_pred_saint = [[0 for x in range(models_J)] for x in range(models_I)]
prob_saint = [[0 for x in range(models_J)] for x in range(models_I)]
for model_index in range(len(models_saint)) :
    model = models_saint[model_index]
    index2d = np.unravel_index(model_index, (models_I, models_J))

    ################################################################################################################
    ### SAINT test set evaluation modified from the classification_scores function in utils.py (from SAINT repo) ###
    model.eval()
    m = torch.nn.Softmax(dim=1)
    # y_test = torch.empty(0).to(device)
    # y_pred = torch.empty(0).to(device)
    # prob = torch.empty(0).to(device)
    with torch.no_grad():
        y_test_i = torch.empty(0).to(device)
        y_pred_i = torch.empty(0).to(device)
        prob_i = torch.empty(0).to(device)
        for i, data in enumerate(testloaders[model_index], 0):
            x_categ, x_cont, y_gts, cat_mask, con_mask = data[0].to(device), data[1].to(device),data[2].to(device),data[3].to(device),data[4].to(device)
            _ , x_categ_enc, x_cont_enc = embed_data_mask(x_categ, x_cont, cat_mask, con_mask,model, vision_dsets[model_index])           
            reps = model.transformer(x_categ_enc, x_cont_enc)
            y_reps = reps[:,0,:]
            y_outs = model.mlpfory(y_reps)
            # import ipdb; ipdb.set_trace()   
            y_test_i = torch.cat([y_test_i,y_gts.squeeze().float()],dim=0)
            y_pred_i = torch.cat([y_pred_i,torch.argmax(y_outs, dim=1).float()],dim=0)
            prob_i = torch.cat([prob_i,m(y_outs)[:,-1].float()],dim=0)
        y_test_saint[index2d[0]][index2d[1]] = y_test_i
        y_pred_saint[index2d[0]][index2d[1]] = y_pred_i
        prob_saint[index2d[0]][index2d[1]] = prob_i
    ################################################################################################################
print("Done.")

# Plot the ROC curve and report the AUC metric for all 12 SAINT models
print("Plotting the ROC curves...")
fig, ax = plt.subplots(models_I, models_J, sharex=True, sharey=True, dpi = 160, figsize=(12, 8.5))
for i in range(models_I) :
    for j in range(models_J) :
        # ROC curve
        fpr, tpr, thresholds = roc_curve(y_test_saint[i][j], prob_saint[i][j])
        ax[i, j].plot(fpr, tpr)
        
        # Formatting
        ax[i, j].plot(np.linspace(0,1, 100), np.linspace(0,1,100),'--r')
        ax[i, j].set_xlabel(('FPR' if i == 2 else ''), fontsize = 12)
        ax[i, j].set_ylabel(('TPR' if j == 0 else ''), fontsize = 12)
        ax[i, j].set(ylim = [0., 1.])
        ax[i, j].set_aspect('equal', 'box')
        ax[i, j].text(0.5, 0.2, f'AUC = {roc_auc_score(y_test_saint[i][j], prob_saint[i][j]):.3f}', fontsize = 12) # AUC metric
        
fig.tight_layout()
print("Done.")

# Save fig.
print("Saving figure...")
plt.savefig("figures/ROC_SAINT.png", dpi = 600)
print("Done.")
print("Done evaluating SAINT.")

##############################
## All models in one figure ##
##############################
print("Plotting the ROC curves for all 60 models in one figure...")
fig, ax = plt.subplots(models_I, models_J, sharex=True, sharey=True, dpi = 160, figsize=(12, 8.5))
fig.tight_layout()
c = plt.rcParams['axes.prop_cycle'].by_key()['color'][0:5]
for i in range(models_I) :
    for j in range(models_J) :
        # Baseline MLPs
        fpr, tpr, thresholds = roc_curve(y_true[i, j], y_pred[i, j]) # ROC curve
        ax[i, j].plot(fpr, tpr)
        
        # XGBoost (Untuned)
        xgb_cl = xgb_classifiers_r[i, j]
        x_test_xgb = XY_xgb_r[i, j][1]
        y_true_xgb = XY_xgb_r[i, j][3]
        y_pred_xgb = xgb_cl.predict_proba(x_test_xgb) # XGB test set predictions
        fpr, tpr, thresholds = roc_curve(y_true_xgb, y_pred_xgb[:,1]) # ROC curve
        ax[i, j].plot(fpr, tpr)
        
        # XGBoost (Tuned)
        xgb_cl_tuned = xgb_classifiers_tuned_r[i, j]
        y_pred_xgb_tuned = xgb_cl_tuned.predict_proba(x_test_xgb) # XGB test set predictions
        fpr, tpr, thresholds = roc_curve(y_true_xgb, y_pred_xgb_tuned[:,1]) # ROC curve
        ax[i, j].plot(fpr, tpr)

        # SAINT
        fpr, tpr, thresholds = roc_curve(y_test_saint[i][j], prob_saint[i][j])
        ax[i, j].plot(fpr, tpr)

        # XGBoost (Tuned, Framingham Only)
        xgb_cl_tunedF = xgb_classifiers_tuned_rF[i, j]
        x_test_xgbF = XY_xgb_rF[i, j][1]
        y_true_xgbF = XY_xgb_rF[i, j][3]
        y_pred_xgb_tunedF = xgb_cl_tunedF.predict_proba(x_test_xgbF) # XGB test set predictions
        fpr, tpr, thresholds = roc_curve(y_true_xgbF, y_pred_xgb_tunedF[:,1]) # ROC curve
        ax[i, j].plot(fpr, tpr)
        
        # Formatting
        ax[i, j].plot(np.linspace(0, 1, 100), np.linspace(0,1,100),'--k')
        ax[i, j].set_xlabel(('FPR' if i == 2 else ''), fontsize = 12)
        ax[i, j].set_ylabel(('TPR' if j == 0 else ''), fontsize = 12)
        ax[i, j].set(ylim = [0., 1.])
        ax[i, j].set_aspect('equal', 'box')
        ax[i, j].text(0.34, 0.30, f'AUC MLP = {roc_auc_score(y_true[i, j], y_pred[i, j]):.3f}', fontsize = 10, color = c[0]) # AUC metric
        ax[i, j].text(0.34, 0.23, f'AUC XGBu = {roc_auc_score(y_true_xgb, y_pred_xgb[:,1]):.3f}', fontsize = 10, color = c[1]) # AUC metric
        ax[i, j].text(0.34, 0.16, f'AUC XGBt = {roc_auc_score(y_true_xgb, y_pred_xgb_tuned[:,1]):.3f}', fontsize = 10, color = c[2]) # AUC metric
        ax[i, j].text(0.34, 0.09, f'AUC SAINT = {roc_auc_score(y_test_saint[i][j], prob_saint[i][j]):.3f}', fontsize = 10, color = c[3]) # AUC metric
        ax[i, j].text(0.34, 0.02, f'AUC XGBtFram = {roc_auc_score(y_true_xgbF, y_pred_xgb_tunedF[:,1]):.3f}', fontsize = 10, color = c[4]) # AUC metric
        ax[i, j].set(xlabel = ('FPR' if i == 2 else ''), ylabel = ('TPR' if j == 0 else ''), ylim = [0., 1.])

fig.tight_layout()
print("Done.")

# Save fig.
print("Saving figure...")
fig.savefig("figures/ROC_Comparison.png", dpi = 600)
print("Done.")

#################################################
## Different evaluation metrics for all models ##
#################################################
# Helper function to convert output probabilities to labels 
def prob_to_labels(y_pred) :
    y_label = np.copy(y_pred)
    y_label[y_label> 0.5] = 1
    y_label[y_label <= 0.5] = 0
    return y_label

print("Computing all evaluation metrics...")
acc = [[0 for x in range(5)] for x in range(N_DATASETS)]
prec = [[0 for x in range(5)] for x in range(N_DATASETS)]
rec = [[0 for x in range(5)] for x in range(N_DATASETS)]
auc = [[0 for x in range(5)] for x in range(N_DATASETS)]
for i in range(models_I) :
    for j in range(models_J) :
        # Baseline MLPs
        y_pred_labels_mlp = prob_to_labels(y_pred[i, j])
        acc_mlp = accuracy_score(y_true[i, j], y_pred_labels_mlp)
        prec_mlp = precision_score(y_true[i, j], y_pred_labels_mlp)
        rec_mlp = recall_score(y_true[i, j], y_pred_labels_mlp)
        auc_mlp = roc_auc_score(y_true[i, j], y_pred[i, j])
        
        # XGBoost (Untuned)
        xgb_cl = xgb_classifiers_r[i, j]
        x_test_xgb = XY_xgb_r[i, j][1]
        y_true_xgb = XY_xgb_r[i, j][3]
        y_pred_xgb = xgb_cl.predict_proba(x_test_xgb) # XGB test set predictions
        y_pred_labels_xgbu = prob_to_labels(y_pred_xgb[:,1])
        acc_xgbu = accuracy_score(y_true_xgb, y_pred_labels_xgbu)
        prec_xgbu = precision_score(y_true_xgb, y_pred_labels_xgbu)
        rec_xgbu = recall_score(y_true_xgb, y_pred_labels_xgbu)
        auc_xgbu = roc_auc_score(y_true_xgb, y_pred_xgb[:,1])
        
        # XGBoost (Tuned)
        xgb_cl_tuned = xgb_classifiers_tuned_r[i, j]
        y_pred_xgb_tuned = xgb_cl_tuned.predict_proba(x_test_xgb) # XGB test set predictions
        y_pred_labels_xgbt = prob_to_labels(y_pred_xgb_tuned[:,1])
        acc_xgbt = accuracy_score(y_true_xgb, y_pred_labels_xgbt)
        prec_xgbt = precision_score(y_true_xgb, y_pred_labels_xgbt)
        rec_xgbt = recall_score(y_true_xgb, y_pred_labels_xgbt)
        auc_xgbt = roc_auc_score(y_true_xgb, y_pred_xgb_tuned[:,1])

        # SAINT
        y_pred_labels_saint = prob_to_labels(prob_saint[i][j])
        acc_saint = accuracy_score(y_test_saint[i][j], y_pred_labels_saint)
        prec_saint = precision_score(y_test_saint[i][j], y_pred_labels_saint)
        rec_saint = recall_score(y_test_saint[i][j], y_pred_labels_saint)
        auc_saint = roc_auc_score(y_test_saint[i][j], prob_saint[i][j])

        # XGBoost (Tuned, Framingham Only)
        xgb_cl_tunedF = xgb_classifiers_tuned_rF[i, j]
        x_test_xgbF = XY_xgb_rF[i, j][1]
        y_true_xgbF = XY_xgb_rF[i, j][3]
        y_pred_xgb_tunedF = xgb_cl_tunedF.predict_proba(x_test_xgbF) # XGB test set predictions
        y_pred_labels_xgbtF = prob_to_labels(y_pred_xgb_tunedF[:,1])
        acc_xgbtF = accuracy_score(y_true_xgbF, y_pred_labels_xgbtF)
        prec_xgbtF = precision_score(y_true_xgbF, y_pred_labels_xgbtF)
        rec_xgbtF = recall_score(y_true_xgbF, y_pred_labels_xgbtF)
        auc_xgbtF = roc_auc_score(y_true_xgbF, y_pred_xgb_tunedF[:,1])

        # Populate the metric arrays
        acc[i * models_J + j][:] = [acc_mlp, acc_xgbu, acc_xgbt, acc_saint, acc_xgbtF]
        prec[i * models_J + j][:] = [prec_mlp, prec_xgbu, prec_xgbt, prec_saint, prec_xgbtF]
        rec[i * models_J + j][:] = [rec_mlp, rec_xgbu, rec_xgbt, rec_saint, rec_xgbtF]
        auc[i * models_J + j][:] = [auc_mlp, auc_xgbu, auc_xgbt, auc_saint, auc_xgbtF]
print("Done.") 

print("Saving all evaluation metrics...")
np.savetxt("evaluation_metrics/accuracy.csv", acc, delimiter = ",")
np.savetxt("evaluation_metrics/precision.csv", prec, delimiter = ",")
np.savetxt("evaluation_metrics/recall.csv", rec, delimiter = ",")
np.savetxt("evaluation_metrics/auc.csv", auc, delimiter = ",")
print("Done.")
