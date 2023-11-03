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

import pickle
######################################

######################################
########## Data processing ###########
######################################
# Preprocesses the dataframe to drop irrelevant columns 
# and extract the rows for a particular sex
def preprocess_df(dataframe, sex = 'both') :
    # If not 'both', extract either female or male examples only
    if sex == 'female' :
        dataframe = dataframe.loc[dataframe['Sex'] == 0]
    elif sex == 'male' :
        dataframe = dataframe.loc[dataframe['Sex'] == 1]
    
    # Drop the irrelevant eid column
    dataframe = dataframe.drop(columns = ['eid']) 
    
    # Drop the superfluous sex feature if one sex is considered
    if not sex == 'both' :
        dataframe = dataframe.drop(columns = ['Sex'])
    
    # Number of patients
    m_patients = len(dataframe.index) 
    
    return dataframe, m_patients

# Builds the feature tensor and the target label vectors (four disease subsets)
def buildXY(dataframe) :
    df = dataframe.copy()
    
    # Extract the target vectors for the four disease subsets
    target_all = df.pop('HeartDisease')
    target_hyper = df.pop('HypertensiveDiseases')
    target_isch = df.pop('IschaemicHeartDiseases')
    target_cond = df.pop('ConductionDisorders')
    targets = [target_all, target_hyper, target_isch, target_cond] # Comprehensive collection
    
    # Convert input features to a tensor for TensorFlow
    features_tensor = tf.convert_to_tensor(df) 
    
    return df, features_tensor, targets

# Generates an oversampled version of the training set assuming a binary classification task. 
# The input x_train and y_train are Pandas dataframes for the features and tragets of the training set, respectively.
def oversample_train(x_train, y_train) :
    # Save column names for dataframe conversion later
    x_train_cols = list(x_train.columns.values)
    
    # Convert to numpy arrays for processing
    x_train = np.array(x_train)
    y_train = np.array(y_train)

    #################
    ### Oversampling
    ### (adapted from: https://www.tensorflow.org/tutorials/structured_data/imbalanced_data)
    #################
    # Boolean mask of positive features
    bool_train_labels = y_train != 0
    
    # Extract positive-class and negative-class features and target labels
    pos_features = x_train[bool_train_labels, :]
    neg_features = x_train[~bool_train_labels, :]
    pos_labels = y_train[bool_train_labels]
    neg_labels = y_train[~bool_train_labels]
    
    # Randomly sample positive feature indices as many times as there are negative features
    ids = np.arange(len(pos_features))
    choices = np.random.choice(ids, len(neg_features))

    # Resample positive features/labels using the randomized indices
    res_pos_features = pos_features[choices]
    res_pos_labels = pos_labels[choices]
    
    # Concatenate resampled positive features/labels and original negative features/labels;
    resampled_features = np.concatenate([res_pos_features, neg_features], axis=0)
    resampled_labels = np.concatenate([res_pos_labels, neg_labels], axis=0)

    # Shuffle the oversampled training set
    order = np.arange(len(resampled_labels))
    np.random.shuffle(order)
    resampled_features = resampled_features[order]
    resampled_labels = resampled_labels[order]
    #################
    
    # Convert back to dataframes and update the XGBoost model structures
    x_train = pd.DataFrame(resampled_features, columns = x_train_cols)
    y_train = pd.DataFrame(resampled_labels, columns = ['Class'])

    return (x_train, y_train)

# Exports the dataset saint_dataset = (x_train, x_val, x_test, y_train, y_val, y_test) to a file
# with name filename. This exported structure can be used directly with the adapted SAINT dataset pipeline.
# This function can be used with saint_dataset that has an oversampled training set.
def save_dataset_saint(filename, saint_dataset) :
    with open(filename, 'wb') as f:
        # Number of examples in each set
        m_train = saint_dataset[0].shape[0]
        m_val = saint_dataset[1].shape[0]
        m_test = saint_dataset[2].shape[0]
        
        # Concatenate the Pandas dataframes
        X_saint = pd.concat([saint_dataset[0], saint_dataset[1], saint_dataset[2]], ignore_index = True)
        m_total = X_saint.shape[0]
        y_saint = pd.concat([saint_dataset[3], saint_dataset[4].to_frame(name = "Class"), saint_dataset[5].to_frame(name = "Class")], ignore_index = True)

        # Build the train, val, and test index arrays for the SAINT dataset pipeline
        train_indices = list(range(0, m_train))
        valid_indices = list(range(m_train, m_train + m_val))
        test_indices = list(range(m_train + m_val, m_total))

        # Save the dataset and index arrays
        pickle.dump((X_saint, y_saint, train_indices, valid_indices, test_indices), f)
######################################