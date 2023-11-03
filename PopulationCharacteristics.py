# This file is designed to be run inside the Research Analysis Platform which is integrated with the UK Biobank dataset

## Import packages
import pyspark
import dxpy
import dxdata
import pandas as pd

# Spark initialization (Done only once; do not rerun this cell unless you select Kernel -> Restart kernel).
sc = pyspark.SparkContext()
spark = pyspark.sql.SparkSession(sc)

# Automatically discover dispensed database name and dataset id
dispensed_database = dxpy.find_one_data_object(
    classname="database", 
    name="app*", 
    folder="/", 
    name_mode="glob", 
    describe=True)
dispensed_database_name = dispensed_database["describe"]["name"]

dispensed_dataset = dxpy.find_one_data_object(
    typename="Dataset", 
    name="app*.dataset", 
    folder="/", 
    name_mode="glob")
dispensed_dataset_id = dispensed_dataset["id"]

dispensed_dataset_id
dataset = dxdata.load_dataset(id=dispensed_dataset_id)
participant = dataset["participant"]

# Returns all field objects for a given UKB showcase field id


def fields_for_id(field_id):
    from distutils.version import LooseVersion
    field_id = str(field_id)
    fields = participant.find_fields(name_regex=r'^p{}(_i\d+)?(_a\d+)?$'.format(field_id))
    return sorted(fields, key=lambda f: LooseVersion(f.name))

# Returns all field names for a given UKB showcase field id


def field_names_for_id(field_id):
    return [f.name for f in fields_for_id(field_id)]


# cardiovascular features and disease diagnoses

# Wall Thickness/Hypertrophic Cardiomyopathy: 2014
field_namesWT = ['eid', 'p31', 'p41270', 'p24124_i2', 'p24125_i2', 'p24126_i2', 'p24127_i2', 'p24128_i2', 'p24129_i2', 'p24130_i2', 'p24131_i2', 'p24132_i2', 'p24133_i2', 
                 'p24134_i2', 'p24135_i2', 'p24136_i2', 'p24137_i2', 'p24138_i2', 'p24139_i2', 'p24140_i2']

# SBP/DBP/Essential Hypertension i0: 2006-2010
field_namesHyp = ['eid', 'p31', 'p41270'] + field_names_for_id("4080_i0") + field_names_for_id("4079_i0")

# PR/1st deg AV block: 2014
field_namesAV = ['eid', 'p31', 'p41270', 'p22330_i2']

# Dilated Cardiomyopathy: 2014
field_namesDCM = ['eid', 'p31', 'p41270', 'p22427_i2', 'p21003_i2', 'p22420_i2', 'p22421_i2']

dfWT = participant.retrieve_fields(names=field_namesWT, engine=dxdata.connect())
dfHyp = participant.retrieve_fields(names=field_namesHyp, engine=dxdata.connect())
dfAV = participant.retrieve_fields(names=field_namesAV, engine=dxdata.connect())
dfDCM = participant.retrieve_fields(names=field_namesDCM, engine=dxdata.connect())

df_WT = dfWT.toPandas()
df_Hyp = dfHyp.toPandas()
df_AV = dfAV.toPandas()
df_DCM = dfDCM.toPandas()

subsetFieldNames = ['p31', 'p24124_i2']
df_WT.dropna(subset=subsetFieldNames, inplace=True)
df_WT.drop(columns=['eid'], inplace=True)
df_WT.reset_index(drop=True, inplace=True)
df_WT.count()
dfWTonly = df_WT.drop(columns=['p31', 'p41270'])
df_WT['WTmax'] = dfWTonly.max(axis=1)
df_WT.drop(columns=['p24124_i2', 'p24125_i2', 'p24126_i2', 'p24127_i2', 'p24128_i2', 'p24129_i2', 'p24130_i2', 'p24131_i2', 'p24132_i2', 'p24133_i2',
                    'p24134_i2', 'p24135_i2', 'p24136_i2', 'p24137_i2', 'p24138_i2', 'p24139_i2', 'p24140_i2'], inplace=True)
df_WT.rename(columns={"p31": "Sex", 'p41270':'ICD10'}, inplace=True)
df_WT.count()

subsetFieldNames = ['p31', 'p4080_i0_a0', 'p4079_i0_a0']
# take the mean of the arrayed features
df_Hyp['SBP'] = df_Hyp[field_names_for_id("4080_i0")].mean(axis=1)
df_Hyp['DBP'] = df_Hyp[field_names_for_id("4079_i0")].mean(axis=1)
df_Hyp.dropna(subset=['SBP', 'DBP'], inplace=True)
df_Hyp.drop(columns=['eid', 'p4080_i0_a0', 'p4080_i0_a1', 'p4079_i0_a0', 'p4079_i0_a1'], inplace=True)
df_Hyp.reset_index(drop=True, inplace=True)
df_Hyp.rename(columns={"p31": "Sex", 'p41270':'ICD10'}, inplace=True)
df_Hyp.count()

subsetFieldNames = ['p31', 'p22330_i2']
df_AV.dropna(subset=subsetFieldNames, inplace=True)
df_AV.reset_index(drop=True, inplace=True)
df_AV.drop(columns=['eid'], inplace=True)
df_AV.rename(columns={"p31": "Sex", 'p41270':'ICD10', 'p22330_i2':'PQ interval'}, inplace=True)
df_AV.count()
df_AV.head()

subsetFieldNames = ['p31', 'p22420_i2', 'p22421_i2', 'p22427_i2', 'p21003_i2']
df_DCM.dropna(subset=subsetFieldNames, inplace=True)
df_DCM.reset_index(drop=True, inplace=True)
df_DCM.drop(columns=['eid'], inplace=True)
df_DCM.rename(columns={"p31": "Sex", 'p41270':'ICD10', 'p22427_i2':'BSA', 'p21003_i2':'Age', 'p22420_i2':'LVEF', 'p22421_i2':'LVEDV'}, inplace=True)
df_DCM.count()

df_WT.to_csv('WallThickness.csv', index=False)
df_Hyp.to_csv('Hypertension.csv', index=False)
df_AV.to_csv('AVBlock.csv', index=False)
df_DCM.to_csv('DCM.csv', index=False)

# %%bash
# dx upload WallThickness.csv --dest
# dx upload Hypertension.csv --dest
# dx upload AVBlock.csv --dest
# dx upload DCM.csv --dest
