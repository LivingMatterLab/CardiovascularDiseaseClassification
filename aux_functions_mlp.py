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

##########################
########## MLP ###########
##########################
# Defines the train/dev/test dataset structure and handling, 
# and initializes the model architecture for MLP baseline training
def build_baseline_MLP(features, target, valtest_frac, batch_size = 32, max_epochs = 100, oversample = False) :
    # Constants for train/val/test set split and 
    m = len(target) # number of patients
    m_validate = int(valtest_frac / 2 * m) # Test size is assumed to be the same as validation size
    train_frac = 1 - valtest_frac
    m_train = int(train_frac * m)
    
    # Buffer size for shuffling
    buffer_size = m
    
    # Extract training set features and targets
    train_features = features[0:m_train, :]
    train_target = target[0:m_train]

    # Build general dataset generator from the feature tensor and the target sequence
    ds = tf.data.Dataset.from_tensor_slices((features, target))
    
    train_ds = [] # Initialize training dataset
    #################
    ### Oversampling in the TensorFlow dataset pipeline
    ### (adapted from: https://www.tensorflow.org/tutorials/structured_data/imbalanced_data)
    if oversample :
        # Boolean mask of positive features
        bool_train_labels = train_target != 0
        
        # Extract positive-class and negative-class features and target labels
        pos_features = tf.boolean_mask(train_features, bool_train_labels)
        neg_features = tf.boolean_mask(train_features, ~bool_train_labels)
        pos_labels = train_target[bool_train_labels]
        neg_labels = train_target[~bool_train_labels]
        
        # Define shuffled dataset generators for the positive and negative classes
        pos_ds = tf.data.Dataset.from_tensor_slices((pos_features, pos_labels)).shuffle(buffer_size).repeat()
        neg_ds = tf.data.Dataset.from_tensor_slices((neg_features, neg_labels)).shuffle(buffer_size).repeat()
        
        # Randomly merge the datasets weights with equalized sampling weights; batch the result
        train_ds = tf.data.Dataset.sample_from_datasets([pos_ds, neg_ds], weights=[0.5, 0.5]).batch(batch_size).prefetch(2)
        # train_ds = tf.data.experimental.sample_from_datasets([pos_ds, neg_ds], weights=[0.5, 0.5]).batch(batch_size).prefetch(2)
        
        # Heuristic choice of steps-per-epoch
        _, neg = np.bincount(target)
        steps_per_epoch = np.ceil(2.0 * neg / batch_size)
    #################
    else :
        # If not oversampling, create a shuffled and batched training set
        train_ds = ds.take(m_train).cache()
        train_ds = train_ds.shuffle(buffer_size, reshuffle_each_iteration = True).repeat().batch(batch_size).prefetch(2)
        steps_per_epoch = m_train // batch_size
    
    
    # Define the validation and test splits 
    validate_ds = ds.skip(m_train).take(m_validate).cache()
    test_ds = ds.skip(m_train).skip(m_validate)

    # Batch the validation set
    validate_ds = validate_ds.batch(batch_size).prefetch(2)
    
    ### Standarization layer defined over the training features
    normalizer = tf.keras.layers.Normalization(axis = -1)
    # normalizer = tf.keras.layers.experimental.preprocessing.Normalization(axis = -1)
    normalizer.adapt(train_features)
    
    # Sequential MLP for binary classification; batch normalization after each hidden layer
    # Each layer with L2-regularization
    model = tf.keras.Sequential([
        normalizer,
        tf.keras.layers.Dense(30, activation = 'relu', kernel_regularizer = tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(30, activation = 'relu', kernel_regularizer = tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(30, activation = 'relu', kernel_regularizer = tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(30, activation = 'relu', kernel_regularizer = tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(30, activation = 'relu', kernel_regularizer = tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(30, activation = 'relu', kernel_regularizer = tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(1, activation = 'sigmoid')
    ])
    
    # Compile the model
    model.compile(optimizer = 'adam',
              loss = tf.keras.losses.BinaryCrossentropy(),
              metrics = [tf.keras.metrics.BinaryAccuracy(name='accuracy'),
                         tf.keras.metrics.Precision(name='precision'),
                         tf.keras.metrics.Recall(name='recall'),
                         tf.keras.metrics.Recall(name='auc')])
    
    return (model, train_ds, validate_ds, test_ds, steps_per_epoch, max_epochs)