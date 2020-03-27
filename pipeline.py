import numpy as np
import pandas as pd
from preprocessor_titanic_2 import Pipeline
import config_titanic as config



pipeline = Pipeline(target = config.TARGET,
					replace = config.REPLACE,
					name = config.NAME,
					categorical_to_impute = config.CATEGORICAL_TO_IMPUTE,
					numerical_cast = config.NUMERICAL_CAST,
					numerical_to_impute = config.NUMERICAL_TO_IMPUTE,
					drop_var=config.DROP_VAR,
					categorical_to_onehot = config.CATEGORICAL_TO_ONEHOT,
					numerical_log = config.NUMERICAL_LOG,
					random_state=0, 
					test_size=0.1,
					percentage = 0.01
					)


if __name__ == '__main__':
	data = pd.read_csv(config.PATH_TO_DATASET)
	pipeline.fit(data)
	pipeline.predict(data)
	pipeline.evaluate()