import re

# to handle datasets
import pandas as pd
import numpy as np

# for visualization
import matplotlib.pyplot as plt

# to divide train and test set
from sklearn.model_selection import train_test_split

# feature scaling
from sklearn.preprocessing import StandardScaler

# to build the models
from sklearn.linear_model import LogisticRegression

# to evaluate the models
from sklearn.metrics import accuracy_score, roc_auc_score

# to persist the model and the scaler
import joblib

# Individual pre-processing and training functions
#==================================================
def load_data(df_path):
    # Function loads data for training
    return pd.read_csv(df_path)

def divide_train_test(df, target):
    # Function divides data set in train and test
    X_train, X_test, y_train, y_test = train_test_split(df,
                                                        df[target],
                                                        test_size=0.1,
                                                        random_state=0)
    return X_train, X_test, y_train, y_test

def replace(df):
	#replace ? to nan values
	df = df.replace('?',np.nan)
	return df


def get_first_cabin(row):
    try:
        return row.split()[0]
    except:
        return np.nan


def get_title(passenger):
    #get the title of the passengers    
    line = passenger
    if re.search('Mrs', line):
        return 'Mrs'
    elif re.search('Mr', line):
        return 'Mr'
    elif re.search('Miss', line):
        return 'Miss'
    elif re.search('Master', line):
        return 'Master'
    else:
        return 'Other'


def cast_numerical(df,var):
    df[var] = df[var].astype('float')
    return df[var]


def impute_numerical(df,var):
    df[var+'_NA'] = np.where(df[var].isnull(), 1, 0)

    # replace NaN by median
    median_val = df[var].median()
    df[var].fillna(median_val, inplace=True)
    return df[var]


def remove_rare_labels(df, var, frequent_labels):
    # groups labels that are not in the frequent list into the umbrella
    # group Rare
    return np.where(df[var].isin(frequent_labels), df[var], 'Rare')


def dummy_variables(df,var):
    df = pd.concat([df,pd.get_dummies(df[var], prefix=var, drop_first=True)], axis=1)
    df.drop(labels=var, axis=1, inplace=True)
    return df


def train_scaler(df, output_path):
    scaler = StandardScaler()
    scaler.fit(df)
    joblib.dump(scaler, output_path)
    return scaler


def scale_features(df, scaler):
    scaler = joblib.load(scaler) # with joblib probably
    return scaler.transform(df)


def train_model(df, target, output_path):
    # initialise the model
    log_model = LogisticRegression(C=0.0005,random_state=0)
    
    # train the model
    log_model.fit(df, target)
    
    # save the model
    joblib.dump(log_model, output_path)
    
    return None


def predict_class(df, model):
    model = joblib.load(model)
    return model.predict(df)

def predict_proba(df,model):
	model = joblib.load(model)
	return model.predict_proba(df)[:,1]
