import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from pathlib import Path
import tarfile
import urllib.request

from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, MinMaxScaler, StandardScaler, FunctionTransformer
from sklearn.compose import TransformedTargetRegressor, ColumnTransformer
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.metrics import root_mean_squared_error

from sklearn.pipeline import Pipeline, make_pipeline

test_set = pd.read_csv('data_kaggle/test.csv')
train_set = pd.read_csv('data_kaggle/train.csv')


features = ['PassengerId', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp','Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']



X = train_set[features].copy()
y= train_set['Surived'].copy()

#print(X.head(10))


#print(train_set.info())
#print(train_set.describe())
#print(train_set.Cabin)

correlation= train_set.corr(numeric_only=True)
print(correlation['Survived'].sort_values(ascending=False))
#Fare had a very strong correlation with surival - Crucial feature, so did Pclass (strong negative correlation)



print(train_set.Name.head(20)) #Some names have Master, some named do not... Indicator? Could length of name also tell us about status?, Lets see...



# every feature has 891 except age and cabin. lets look into those
features = ('PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',
       'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked')

