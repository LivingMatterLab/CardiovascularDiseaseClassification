## Import packages
import pyspark
import dxpy
import dxdata

# This file is designed to be run inside the Research Analysis Platform which is integrated with the UK Biobank dataset

### Load Biobank dataset with Spark SQL

# Spark initialization
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


### Get cardiovascular features and diagnoses


def fields_for_id(field_id):  # Returns all field objects for a given UKB showcase field id
    from distutils.version import LooseVersion
    field_id = str(field_id)
    fields = participant.find_fields(name_regex=r'^p{}(_i\d+)?(_a\d+)?$'.format(field_id))
    return sorted(fields, key=lambda f: LooseVersion(f.name))


def field_names_for_id(field_id):  # Returns all field names for a given UKB showcase field id
    return [f.name for f in fields_for_id(field_id)]


field_names = ['eid', 'p31', 'p2443_i2', 'p20116_i2', 'p30760_i0', 'p30690_i0', 'p21001_i2', 'p41270', 'p21003_i2',
               'p12336_i2', 'p12338_i2', 'p22334_i2', 'p22330_i2', 'p22338_i2', 'p12340_i2', 'p22331_i2', 'p22332_i2',
               'p22333_i2', 'p22335_i2', 'p22336_i2', 'p22337_i2', 'p22672_i2', 'p22675_i2', 'p22678_i2', 'p22681_i2',
               'p22671_i2', 'p22674_i2', 'p22677_i2',
               'p22680_i2', 'p22670_i2', 'p22673_i2', 'p22676_i2', 'p22679_i2', 'p4199_i2', 'p4198_i2', 'p4200_i2',
               'p4194_i2', 'p21021_i2', 'p4196_i2',
               'p4195_i2', 'p22426_i2', 'p22425_i2', 'p22424_i2', 'p22420_i2', 'p22421_i2', 'p22422_i2', 'p22423_i2'] + \
              field_names_for_id("12681_i2") + field_names_for_id("12680_i2") + field_names_for_id(
    "12678_i2") + field_names_for_id("12677_i2") + field_names_for_id("12675_i2") + \
              field_names_for_id("12683_i2") + field_names_for_id("12684_i2") + field_names_for_id(
    "12687_i2") + field_names_for_id("12679_i2") + field_names_for_id("12676_i2") + \
              field_names_for_id("12686_i2") + field_names_for_id("12674_i2")

df = participant.retrieve_fields(names=field_names, engine=dxdata.connect())

# Convert to Pandas dataframe and drop missing values
df_pandas = df.toPandas()

subsetFieldNames = ['p31', 'p2443_i2', 'p20116_i2', 'p30760_i0', 'p30690_i0', 'p21001_i2', 'p21003_i2', 'p12336_i2',
                    'p12338_i2', 'p22334_i2', 'p22330_i2', 'p22338_i2', 'p12340_i2', 'p22331_i2', 'p22332_i2',
                    'p22333_i2', 'p22335_i2', 'p22336_i2', 'p22337_i2', 'p4199_i2', 'p4198_i2', 'p4200_i2', 'p4194_i2',
                    'p21021_i2', 'p4196_i2',
                    'p4195_i2', 'p22426_i2', 'p22425_i2', 'p22424_i2', 'p22420_i2', 'p22421_i2', 'p22422_i2',
                    'p22423_i2',
                    'p22672_i2', 'p22675_i2', 'p22678_i2', 'p22681_i2', 'p22671_i2', 'p22674_i2', 'p22677_i2',
                    'p22680_i2', 'p22670_i2', 'p22673_i2',
                    'p22676_i2', 'p22679_i2']

df_pandas.dropna(subset=subsetFieldNames, inplace=True)
df_pandas.reset_index(drop=True, inplace=True)

# Visualize the number of participants in the dataset after dropping the non-arrayed missing values
df_pandas.count()

# take the mean of the arrayed features
df_pandas['12681avg'] = df_pandas[field_names_for_id("12681_i2")].mean(axis=1)
df_pandas['12680avg'] = df_pandas[field_names_for_id("12680_i2")].mean(axis=1)
df_pandas['12678avg'] = df_pandas[field_names_for_id("12678_i2")].mean(axis=1)
df_pandas['12677avg'] = df_pandas[field_names_for_id("12677_i2")].mean(axis=1)
df_pandas['12675avg'] = df_pandas[field_names_for_id("12675_i2")].mean(axis=1)
df_pandas['12683avg'] = df_pandas[field_names_for_id("12683_i2")].mean(axis=1)
df_pandas['12684avg'] = df_pandas[field_names_for_id("12684_i2")].mean(axis=1)
df_pandas['12687avg'] = df_pandas[field_names_for_id("12687_i2")].mean(axis=1)
df_pandas['12679avg'] = df_pandas[field_names_for_id("12679_i2")].mean(axis=1)
df_pandas['12676avg'] = df_pandas[field_names_for_id("12676_i2")].mean(axis=1)
df_pandas['12686avg'] = df_pandas[field_names_for_id("12686_i2")].mean(axis=1)
df_pandas['12674avg'] = df_pandas[field_names_for_id("12674_i2")].mean(axis=1)

# drop any participants with missing features
df_pandas.dropna(
    subset=['12681avg', '12680avg', '12678avg', '12677avg', '12675avg', '12683avg', '12684avg', '12687avg', '12679avg',
            '12676avg', '12686avg', '12674avg'], inplace=True)
df_pandas.reset_index(drop=True, inplace=True)

# final number of participants in dataset
df_pandas['eid'].count()

# drop columns of arrayed features, keep only the average
namesDrop = field_names_for_id("12681_i2") + field_names_for_id("12680_i2") + field_names_for_id(
    "12678_i2") + field_names_for_id("12677_i2") + field_names_for_id("12675_i2") + field_names_for_id(
    "12683_i2") + field_names_for_id("12684_i2") + field_names_for_id("12687_i2") + field_names_for_id(
    "12679_i2") + field_names_for_id("12676_i2") + field_names_for_id("12686_i2") + field_names_for_id("12674_i2")
df_pandas.drop(columns=namesDrop, inplace=True)

# save to csv
df_pandas.to_csv('CardioPhenoFraminghamPlus.csv', index=False)

# %%bash
# dx upload CardioPhenoFraminghamPlus.csv --dest /
