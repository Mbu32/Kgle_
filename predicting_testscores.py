import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from pathlib import Path
import tarfile
import urllib.request
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, GridSearchCV, RandomizedSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler, FunctionTransformer, PolynomialFeatures
from sklearn.compose import TransformedTargetRegressor, ColumnTransformer
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.metrics import root_mean_squared_error
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.svm import LinearSVR, SVR


#most likely, I have a feeling the best way to calculate this would be through a Decision tree method. Maybe Kneighbour?
#could even be a regression tbh
train = pd.read_csv('data_playground/train.csv')
test = pd.read_csv('data_playground/test.csv')

print(train.info())

'''
 #   Column            Non-Null Count   Dtype  
---  ------            --------------   -----  
 0   id                630000 non-null  int64  
 1   age               630000 non-null  int64  
 2   gender            630000 non-null  object 
 3   course            630000 non-null  object
 4   study_hours       630000 non-null  float64
 5   class_attendance  630000 non-null  float64
 6   internet_access   630000 non-null  object
 7   sleep_hours       630000 non-null  float64
 8   sleep_quality     630000 non-null  object
 9   study_method      630000 non-null  object
 10  facility_rating   630000 non-null  object
 11  exam_difficulty   630000 non-null  object
 12  exam_score        630000 non-null  float64
'''

features = [ 'age', 'gender', 'course', 'study_hours', 'class_attendance',
       'internet_access', 'sleep_hours', 'sleep_quality', 'study_method', 
       'facility_rating', 'exam_difficulty']

label = 'exam_score'

cat_features = ['gender','course','exam_difficulty','internet_access','sleep_quality']

X = train[features]
y = train[label]
ids = train['id']




'''FEATURE ENGINEERING'''
#Interactions terms = Sleep_hours x Sleep_quality ... study_hours x internet access .... class attendance x course

sleep_q_map = {'poor':1,'average':2,'good':3}
X['sleep_q_encoded'] = X['sleep_quality'].map(sleep_q_map)
print(X['sleep_quality'].value_counts())

X['sleep_qualityxhours'] = X['sleep_q_encoded']*X['sleep_hours']


# 2
internet_access_map = {'yes':2,"no":1}
X['internet_encoded'] = X['internet_access'].map(internet_access_map)

X['studhours_internet'] = X['internet_encoded']*X['study_hours']

#3
