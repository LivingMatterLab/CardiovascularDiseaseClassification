# Cardiovascular Disease Classification
## CS230 Deep Learning Project

**Note**: This repository contains only the code, not the data. Researchers can [apply](https://www.ukbiobank.ac.uk/enable-your-research/apply-for-access) to access the UK Biobank to complete health-related research that is in the public interest.

## Our contribution
We used the UK Biobank data to train sex-specific classifiers of cardiovascular disease. Three different model types were evaluated: MLP (baseline), XGBoost, and SAINT. The SAINT implementation is adapted from the article [SAINT: Improved Neural Networks for Tabular Data via Row Attention and Contrastive Pre-Training](https://arxiv.org/abs/2106.01342) with the corresponding repository: [Saint GitHub](https://github.com/somepago/saint).

We implemented the following scripts:
- `PopulationCharacteristics.py`
- `ScatterPlots.py`
- `CardioPhenoBiobank.py`
- `CardioPhenoExtract.py`
- `preprocess_datasets.py`
- `train_mlp_models.py`
- `train_xgb_models.py`
- `build_saint_datasets.py`
- `evaluate_all.py`
- `shap_analysis.py`
- `cross_evaluation.py`
- `aux_functions_data.py`
- `aux_functions_mlp.py`
- `aux_functions_xgb.py`

The following scripts were modified from the original [Saint repository](https://github.com/somepago/saint):
- `SAINT/train_robust.py`
- `SAINT/data_openml.py`

Our main adjustments to the SAINT implementation allow the use of custom pickled tabular datasets (with explicit train-val-test splits, and training set oversampling) and to streamline the evaluation pipeline. The remaining files in the `SAINT` directory have not been modified from the original [Saint repository](https://github.com/somepago/saint).

## Environment
We provide an `environment.yml` file for use with `miniconda` or `anaconda`. You can create the environment required for executing the code by running

```
conda env create -f environment.yml
```

The code was executed on Ubuntu 22.04.2 LTS using the conda environment defined by `environment.yml`.

## Provided model weights
We provide the model weights obtained for all 60 trained classifiers:
- directory `models_MLP` contains trained MLP model weights
- directory `models_XGB` contains trained XGBoost (both untuned and tuned) ensembles
- directory `models_XGB_Fram` contains XGBoost ensembles trained and tuned on Framingham-score features only
- directory `bestmodels_saint` contains trained SAINT model weights


## Analysis for sex differences in cardiovascular disease diagnosis

`PopulationCharacteristics.py` extracts hypertension, first degree AV block, and dilated and hypertrophic cardiomyopathy information using the Research Analysis Platform integrated with the UK Biobank database.

`ScatterPlots.py` processes the data returned from `PopulationCharacteristics.py` to produce violin and scatter plots of the data. The Teichholz formula is implemented to converted left ventricle end-diastolic volume to diameter measurments.

## Data preparation

`CardioPhenoBiobank.py` extracts the cardiovascular features and disease diagnoses we selected from the entire UK Biobank database. Spark SQL is used to gather the features and then after converting to a Pandas dataframe we remove missing values and consolidate any arrayed features into one column, e.g. taking the mean of four consecutive blood pressure measurements.

`CardioPhenoExtract.py` takes in pre-filtered UK Biobank data and adds a column indicating whether a person has been diagnosed with cardiovascular disease (1) or not (0). Diagnosis is based on ICD10 codes. Smoking and diabetes status are also simplified to a binary representation with a (1) if diagnosed with diabetes or a current/previous smoker and (0) if else, e.g. participant selected "prefer not to answer". A spreadsheet showing the count by sex for the four cardiovascular disease variants is also generated.

## Dataset pre-processing

`preprocess_datasets.py` preprocesses the datasets with the following steps: 
- Shuffle the overall dataset for randomness. Shuffling at the pre-processing stage (instead of on a per-model basis) guarantees that every model will be evaluated on the same test set, regardless of the differences in the utilized dataset pipeline.
- Extract the 12 different datasets:
  - Both sexes, Any disease
  - Both sexes, Hypertension
  - Both sexes, Ischemic disease
  - Both sexes, Conduction disorder
  - Female only, Any disease
  - Female only, Hypertension
  - Female only, Ischemic disease
  - Female only, Conduction disorder
  - Male only, Any disease
  - Male only, Hypertension
  - Male only, Ischemic disease
  - Male only, Conduction disorder.
- Build iterable dataset collections for streamlined training, hyperparameter tuning, and performance evaluation.
- Pickle the data sets

`build_saint_datasets.py` builds and saves the datasets in a format that is readily accessible for the SAINT input pipeline. Performs train-val-test splits and applies oversampling to the training set before exporting the datasets.

`aux_functions_data.py` implements a library of auxiliary functions for data processing, training set oversampling, and export. 

## Classifier training
### MLP training

`train_mlp_models.py` performs the following steps:
- Builds the MLP baseline models
- Trains the MLP baseline models
- Saves the MLP baseline models
The **Dataset pre-processing** scripts have to be executed first. Includes an oversampling step for the training set data.

`aux_functions_mlp.py` implements a library of auxiliary functions for building the MLP models. 

### XGBoost training

`train_xgb_models.py` performs the following steps:
- Initializes the XGBoost ensembles
- Trains the XGBoost ensembles
- Performs random-search hyperparameter tuning for each XGBoost ensemble. The implementation is parallelized on the CPU.
- Saves both the untuned and the tuned XGBoost ensembles.
The **Dataset pre-processing** scripts have to be executed first. Includes an oversampling step for the training set data.

`aux_functions_xgb.py` implements a library of auxiliary functions for training and tuning the XGBoost ensembles.

### SAINT training

`SAINT/train_robust.py` trains the SAINT model given a pickled dataset (using `build_saint_datasets.py`) using contrastive pretraining and intersample attention. This function is modified from the [Saint repository](https://github.com/somepago/saint).

`SAINT/data_openml.py` implements a library of auxiliary functions for importing the dataset for SAINT training. It was adapted from the [Saint repository](https://github.com/somepago/saint) to allow pickled dataset input, as opposed to OpenML datasets only in the original implementation. 

## Classifier evaluation

`evaluate_all.py` performs a comprehensive evaluation of all 60 classifiers on the corresponding 12 test sets. Generates ROC curves and computes the AUC metric for all 60 classifiers.  

## SHAP analysis

`shap_analysis.py` runs a SHAP analysis of feature importance for the 12 tuned XGBoost models with all input features. Generates the corresponding 12 SHAP summary figures.

## Classifier cross-evaluation

`cross_evaluation.py` performs a cross-evaluation AUC-ROC performance analysis in which the XGBoost models trained on one dataset are evaluated on test sets from other datasets. For example, the model trained on the BA dataset (both sexes, any disease) is cross-evaluated on the FA test set (female only, any disease).
