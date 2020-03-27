import preprocess_titanic as pf
import config_titanic as config

def predict(data, Class_type = True):
	# replace '?' with nan
	data = pf.replace(data)

	# get first cabin
	data['cabin'] = data['cabin'].apply(pf.get_first_cabin)

	# get title
	data['title'] = data[config.NAME].apply(pf.get_title)

	# cast numerical variables into float
	for var in config.NUMERICAL_CAST :
		data[var] = pf.cast_numerical(data,var)

	# impute numerical missing values
	for var in config.NUMERICAL_TO_IMPUTE :
		data[var] = pf.impute_numerical(data,var)

	# Group rare labels
	for var in config.CATEGORICAL_TO_ONEHOT:
	    data[var] = pf.remove_rare_labels(data, var, config.FREQUENT_LABELS[var])

	# crate dummy variables
	data = pf.dummy_variables(data,config.CATEGORICAL_TO_ONEHOT)


	# drop unnecessary features
	data = data.drop(columns = config.DROP_VAR)


	data['embarked_Rare'] = 0


	# scale variables
	data = pf.scale_features(data, config.OUTPUT_SCALER_PATH)


	if Class_type:
    # make predictions
		predictions = pf.predict_class(data, config.OUTPUT_MODEL_PATH)
	else :
		predictions = pf.predict_proba(data,config.OUTPUT_MODEL_PATH)

	return predictions

# ======================================
    
# small test that scripts are working ok
    
if __name__ == '__main__':
    
    from math import sqrt
    import numpy as np
    
    # to evaluate the models
    from sklearn.metrics import accuracy_score, roc_auc_score
    
    import warnings
    warnings.simplefilter(action='ignore')
    
    # Load data
    data = pf.load_data(config.PATH_TO_DATASET)
    X_train, X_test, y_train, y_test = pf.divide_train_test(data,
                                                            config.TARGET)
    
    pred_class = predict(X_test)
    pred_proba = predict(X_test,Class_type=False)
    

    # determine mse and rmse
    print('test roc-auc: {}'.format(roc_auc_score(y_test, pred_proba)))
    print('test accuracy: {}'.format(accuracy_score(y_test, pred_class)))
    print()
        
        