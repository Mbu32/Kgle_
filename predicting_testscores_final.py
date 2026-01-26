import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import os
import urllib.request
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler, FunctionTransformer, PolynomialFeatures
from sklearn.compose import TransformedTargetRegressor, ColumnTransformer, make_column_selector, make_column_transformer
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics import root_mean_squared_error
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.model_selection import cross_val_score, cross_val_predict, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.svm import LinearSVR, SVR
from sklearn.ensemble import VotingRegressor, StackingRegressor, RandomForestRegressor, ExtraTreesRegressor,BaggingRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression, Ridge,Lasso,ElasticNet
from sklearn.svm import LinearSVR, SVR
from sklearn.inspection import permutation_importance
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from perm_class import RegressionCV
import time
from scipy.stats import loguniform, uniform


start_time = time.time()


train = pd.read_csv('data_playground/train.csv')
test_set = pd.read_csv('data_playground/test.csv')

features = [ 'age', 'gender', 'course', 'study_hours', 'class_attendance',
       'internet_access', 'sleep_hours', 'sleep_quality', 'study_method', 
       'facility_rating', 'exam_difficulty']

label = 'exam_score'


X = train[features].copy()
y = train[label].copy()

X_train, X_val , y_train, y_val = train_test_split(X,y,test_size=.2,random_state=42)


test_set_1 = test_set[features].copy()





catboost = CatBoostRegressor(random_state=42,
                             verbose=0,loss_function='RMSE',
                             eval_metric='RMSE',
                             cat_features=['gender','course','exam_difficulty',
                                                          'study_method','facility_rating',
                                                          'sleep_quality','internet_access'],
                            iterations=300,
                            subsample=.8,
                            learning_rate=.2,
                            depth=8,
                            colsample_bylevel=.6,
                            l2_leaf_reg=7,
                            min_data_in_leaf=20)



"""submissions = pd.DataFrame({
    'id':test_set['id'],
    'exam_score': prediction
})

submissions.to_csv('Submissions_csv/testscores_submission1.csv',index=False)"""




elapsed = time.time() - start_time
hours = int(elapsed // 3600)
minutes = int((elapsed % 3600) // 60)
seconds = int(elapsed % 60)

print(f"\n⏱️  Total time: {hours:02d}:{minutes:02d}:{seconds:02d}")
os.system(f'say "Program finished in {hours} hours, {minutes} minutes, {seconds} seconds"')