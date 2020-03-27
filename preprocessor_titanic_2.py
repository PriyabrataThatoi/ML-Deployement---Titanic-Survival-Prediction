import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
import joblib


class Pipeline:

	def __init__(self,target,replace,name,categorical_to_impute,
				numerical_cast,numerical_to_impute,drop_var,categorical_to_onehot,
				numerical_log, random_state=0, test_size=0.1,percentage = 0.01):

		#data
		self.X_train = None
		self.y_train = None
		self.X_test = None
		self.y_test = None

		#variables
		self.target = target
		self.replace = replace
		self.name = name
		self.categorical_to_impute = categorical_to_impute
		self.numerical_cast = numerical_cast
		self.numerical_to_impute = numerical_to_impute
		self.drop_var = drop_var
		self.categorical_to_onehot = categorical_to_onehot
		self.numerical_log = numerical_log

		#models
		self.model = LogisticRegression(C=0.0005,random_state=0)
		self.scalar = StandardScaler()

		#parameters
		self.random_state = random_state
		self.test_size = test_size
		self.percentage = percentage

		#engineering parameters (to be learnt from data)
		self.imputing_dict = {}
		self.frequent_category_dict = {}



	# Individual pre-processing and training functions
	#==================================================

	def find_replacement_variable(self):

		for variable in self.numerical_to_impute:
			replacement = self.X_train[variable].mode()[0]
			self.imputing_dict[variable] = replacement
		return self

	def find_frequent_labels(self):

		for variable in self.categorical_to_onehot:
			tmp = self.X_train.groupby(variable)[self.target].count()/len(self.X_train)
			self.frequent_category_dict[variable] = tmp[tmp > self.percentage].index
		return self


	# Functions to transform data
	#==================================

	def replace(self,df):
		#replace ? to nan values
		df = df.copy()
		df = df.replace(self.replace,np.nan)
		return df


	def get_first_cabin(self,row):
	    try:
	        return row.split()[0]
	    except:
	        return np.nan


	def get_title(self,passenger):
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


	def cast_numerical(self,df):
		df=df.copy()
		for variable in self.numerical_cast:
			df[variable] = df[variable].astype('float')
		return df


	def impute_numerical(self,df):
		df=df.copy()
		for var in self.numerical_to_impute:
			df[variable+'_NA'] = np.where(df[variable].isnull(), 1, 0)
	    
			median_val = find_replacement_variable(variable)
			df[variable].fillna(median_val, inplace=True)
		return df


	def remove_rare_labels(self,df):
	    # groups labels that are not in the frequent list into the umbrella
	    # group Rare
	    df=df.copy()
	    for variable in self.categorical_to_onehot:
	    	df = np.where(df[variable].isin(frequent_category_dict[variable]),df[variable], 'Rare')
	    return df


	def dummy_variables(self,df):
		df= df.copy()
		df = pd.concat([df,pd.get_dummies(df[self.categorical_to_onehot],
									     prefix=self.categorical_to_impute, 
									     drop_first=True)],
									     axis=1)
		df.drop(labels=self.drop_var, axis=1, inplace=True)
		return df

# ====   master function that orchestrates feature engineering =====

	def fit(self,data):
		'''pipeline to learn parameters from data, fit the scaler and logistic regression'''
        # setarate data sets
		self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
			data, data[self.target],
			test_size = self.test_size,
			random_state = self.random_state)
        
		self.find_replacement_variable()
		self.find_frequent_labels()

        # replace '?' with nan
		self.X_train = self.replace(self.X_train)
		self.X_test = self.replace(self.X_test)
		# get first cabin
		self.X_train['cabin'] = self.X_train['cabin'].apply(self.get_first_cabin)
		self.X_test['cabin'] = self.X_test['cabin'].apply(self.get_first_cabin)

		# get title
		self.X_train['title'] = self.X_train[self.NAME].apply(self.get_title)
		self.X_test['title'] = self.X_test[self.NAME].apply(self.get_title)

		# cast numerical variables into float
		self.X_train = self.cast_numerical(self.X_train)
		self.X_test = self.cast_numerical(self.X_test)

		# impute numerical missing values
		self.X_train = self.impute_numerical(self.X_train)
		self.X_test = self.impute_numerical(self.X_test)

		# Group rare labels
		self.X_train = self.remove_rare_labels(self.X_train)
		self.X_test = self.remove_rare_labels(self.X_test)

		# crate dummy variables
		self.X_train = self.dummy_variables(self.X_train)
		self.X_test = self.dummy_variables(self.X_test)

		# drop unnecessary features
		self.X_train = self.X_train.drop(columns = self.DROP_VAR)
		self.X_test = self.X_test.drop(columns = self.DROP_VAR)
		self.X_test['embarked_Rare'] = 0

		# train scaler and save
		self.scaler.fit(self.X_train)

		# scale train set
		self.X_train = self.scaler.transform(self.X_train)
		self.X_test = self.scaler.transform(self.X_test)

		# train model and save
		self.model.fit(self.X_train,self.y_train)

		return self
	
	def predict(self,data):
		predictions = self.model.predict(data)
		return predictions

	def evaluate_model(self):
		pred_class = self.model.predict(self.X_train)
		pred_proba = self.model.predict_proba(self.X_train)

		print('test roc-auc: {}'.format(roc_auc_score(self.y_train, pred_proba)))
		print('test accuracy: {}'.format(accuracy_score(self.y_train, pred_class)))
		print()