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


###################
###### Data #######
###################
# Import datasets
print("Loading data...")
csv_file = 'CardioDiseaseFraminghamPlus.csv'
csv_fileFram = 'FraminghamOnly.csv'
dataframe = pd.read_csv(csv_file)
dataframeFram = pd.read_csv(csv_fileFram)
print("Done.")

# Shuffle the data frame for randomness
print("Shuffling data...")
dataframe = dataframe.sample(frac = 1, random_state = 2).reset_index(drop = True) 
dataframeFram = dataframeFram.sample(frac = 1, random_state = 2).reset_index(drop = True)
print("Done.")

print("Pre-processing data (All features). Generating 12 datasets...")
# Pre-process the Cardiovascular features dataframe for both-sexes, female-only, and male-only datasets
df_both, m_both = preprocess_df(dataframe)
df_female, m_female = preprocess_df(dataframe, sex = 'female')
df_male, m_male = preprocess_df(dataframe, sex = 'male')

# Extract the feature tensors and target vectors (1 target vector per disease subset) 
# for the three dataframes
features_both_df, features_both, targets_both = buildXY(df_both)
features_female_df, features_female, targets_female = buildXY(df_female)
features_male_df, features_male, targets_male = buildXY(df_male)

# Build the dataset collections
features_df = [features_both_df, features_female_df, features_male_df]
features = [features_both, features_female, features_male]
targets = [targets_both, targets_female, targets_male]
print("Done.")

print("Pre-processing data (Framingham only). Generating 12 datasets...")
# Pre-process the Framingham only dataframe for both-sexes, female-only, and male-only datasets
df_bothF, m_bothF = preprocess_df(dataframeFram)
df_femaleF, m_femaleF = preprocess_df(dataframeFram, sex = 'female')
df_maleF, m_maleF = preprocess_df(dataframeFram, sex = 'male')

# Extract the feature tensors and target vectors (1 target vector per disease subset) 
# for the three dataframes
features_both_dfF, features_bothF, targets_bothF = buildXY(df_bothF)
features_female_dfF, features_femaleF, targets_femaleF = buildXY(df_femaleF)
features_male_dfF, features_maleF, targets_maleF = buildXY(df_maleF)

# Build the dataset collections
features_dfF = [features_both_dfF, features_female_dfF, features_male_dfF]
featuresF = [features_bothF, features_femaleF, features_maleF]
targetsF = [targets_bothF, targets_femaleF, targets_maleF]
print("Done.")

# Save all datasets (pickle)
print("Saving all datasets...")
with open('preprocessed_datasets/data_all.pkl', 'wb') as f:
    pickle.dump((features_df, features, targets), f)
with open('preprocessed_datasets/data_F.pkl', 'wb') as f:
    pickle.dump((features_dfF, featuresF, targetsF), f)
print("Done.")

# # Plot pairwise relationships of selected features
# feature_plot = sns.pairplot(df_both.iloc[:, np.r_[-4, 0:8]], hue = 'HeartDisease', diag_kind='kde');

# # Fig. save
# fig = feature_plot.fig
# fig.savefig("PairwisePlot.png", dpi = 600) 