# ====   PATHS ===================

PATH_TO_DATASET = "https://www.openml.org/data/get_csv/16826755/phpMYEkMl"

# ======= PARAMETERS ===============

# variable groups for engineering steps
TARGET = 'survived'

REPLACE = '?'

NAME = 'name'

CATEGORICAL_TO_IMPUTE = ['cabin','embarked']

NUMERICAL_CAST = ['age','fare']

NUMERICAL_TO_IMPUTE =['age','fare']

DROP_VAR =['name','ticket', 'boat', 'body','home.dest']

CATEGORICAL_TO_ONEHOT = ['sex','cabin','embarked','title']

# variables to transofmr
NUMERICAL_LOG = ['age','fare']
