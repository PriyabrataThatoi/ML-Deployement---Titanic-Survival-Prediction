import numpy as np

import preprocess_titanic as pf
import config_titanic as config

import warnings
warnings.simplefilter(action='ignore')

# ================================================
# TRAINING STEP - IMPORTANT TO PERPETUATE THE MODEL

# Load data
data = pf.load_data(config.PATH_TO_DATASET)

# divide data set
X_train, X_test, y_train, y_test = pf.divide_train_test(data, config.TARGET)

# replace '?' with nan
X_train = pf.replace(X_train)

# get first cabin
X_train['cabin'] = X_train['cabin'].apply(pf.get_first_cabin)

# get title
X_train['title'] = X_train[config.NAME].apply(pf.get_title)

# cast numerical variables into float
for var in config.NUMERICAL_CAST :
	X_train[var] = pf.cast_numerical(X_train,var)

# impute numerical missing values
for var in config.NUMERICAL_TO_IMPUTE :
	X_train[var] = pf.impute_numerical(X_train,var)

# Group rare labels
for var in config.CATEGORICAL_TO_ONEHOT:
    X_train[var] = pf.remove_rare_labels(X_train, var, config.FREQUENT_LABELS[var])

# crate dummy variables
X_train = pf.dummy_variables(X_train,config.CATEGORICAL_TO_ONEHOT)


# drop unnecessary features
X_train = X_train.drop(columns = config.DROP_VAR)


# train scaler and save
scaler = pf.train_scaler(X_train,
                         config.OUTPUT_SCALER_PATH)

# scale train set
X_train = scaler.transform(X_train)

# train model and save
pf.train_model(X_train,
               y_train,
               config.OUTPUT_MODEL_PATH)

print('Finished training')

