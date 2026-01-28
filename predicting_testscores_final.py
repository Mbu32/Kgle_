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
from perm_class import RegressionCV, RegressionCV1
import time
from scipy.stats import loguniform, uniform, randint

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











def poly_features_single(X):
    return (X.iloc[:, 0] ** 2).values.reshape(-1, 1)

def log_features_single(X):
    return (np.log1p(X.iloc[:, 0])).values.reshape(-1, 1)

def sqrt_single(X):
    return (np.sqrt(X.iloc[:, 0])).values.reshape(-1, 1)

def interaction_term(X):
    return (X.iloc[:, 0] * X.iloc[:, 1]).values.reshape(-1, 1)  

def ratio_term(X):
    eps = 1e-5
    return (X.iloc[:, 0] / (X.iloc[:, 1] + eps)).values.reshape(-1, 1)


cat_pipeline = make_pipeline(
    SimpleImputer(strategy='most_frequent'),
    OneHotEncoder(handle_unknown='ignore'),
    #StandardScaler(with_mean=False)
)

cat_cols =['internet_access','gender','course','exam_difficulty','study_method','facility_rating','sleep_quality']


preprocessing_numeric = ColumnTransformer([
    
    ('core', make_pipeline(SimpleImputer(strategy='median')), ['study_hours', 'class_attendance', 'sleep_hours', 'age']),


    ('poly_study', FunctionTransformer(poly_features_single, validate=False), ['study_hours']),
    ('poly_attendance', FunctionTransformer(poly_features_single, validate=False), ['class_attendance']),
    ('poly_sleep', FunctionTransformer(poly_features_single, validate=False), ['sleep_hours']),
    ('poly_age', FunctionTransformer(poly_features_single, validate=False), ['age']),
    
    ('log_study', FunctionTransformer(log_features_single, validate=False), ['study_hours']),
    ('log_attendance', FunctionTransformer(log_features_single, validate=False), ['class_attendance']),
    ('log_sleep', FunctionTransformer(log_features_single, validate=False), ['sleep_hours']),
    
    ('sqrt_study', FunctionTransformer(sqrt_single, validate=False), ['study_hours']),
    ('sqrt_attendance', FunctionTransformer(sqrt_single, validate=False), ['class_attendance']),
    
    ('interaction_att_study', FunctionTransformer(interaction_term, validate=False), ['class_attendance', 'study_hours']),
    ('interaction_sleep_study', FunctionTransformer(interaction_term, validate=False), ['sleep_hours', 'study_hours']),
    ('interaction_att_sleep', FunctionTransformer(interaction_term, validate=False), ['class_attendance', 'sleep_hours']),
    ('interaction_age_study', FunctionTransformer(interaction_term, validate=False), ['age', 'study_hours']),
    
    ('ratio_study_sleep', FunctionTransformer(ratio_term, validate=False), ['study_hours', 'sleep_hours']),
    ('ratio_att_sleep', FunctionTransformer(ratio_term, validate=False), ['class_attendance', 'sleep_hours']),
    ('ratio_att_study', FunctionTransformer(ratio_term, validate=False), ['class_attendance', 'study_hours']),
    ('cat_xgb', cat_pipeline,cat_cols)


], remainder='passthrough',sparse_threshold=0)  

lin_pipeline = Pipeline([
    ('preprocessing',preprocessing_numeric),
    ('scaler',StandardScaler()),
    ('model',Ridge())
])



lin_map = {
    'model__alpha': np.logspace(-4, 4, 20),  # 0.0001 to 10000
    'model__solver': ['auto', 'svd', 'cholesky'], #, 'lsqr', 'saga'
    'model__max_iter': [1000, 2000, 5000],
    'model__tol': [1e-3, 1e-4, 1e-5],  # Convergence tolerance
}


rv = RandomizedSearchCV(lin_pipeline,lin_map,
                        cv=3,random_state=42,
                        n_iter=15,
                        scoring='neg_root_mean_squared_error',
                        n_jobs=-1)


rv.fit(X_train,y_train)
print(f'the best params: {rv.best_params_}')
print(f'the best scores: {-rv.best_score_:.4f}')




"""lin_pipeline.fit(X_train,y_train)
y_pred = lin_pipeline.predict(X_val)

rmse_score = root_mean_squared_error(y_val,y_pred)
print(rmse_score)"""







xgb_pipeline = Pipeline([('preprocessing',preprocessing_numeric),
                         ('model',XGBRegressor(learning_rate=0.046415888336127774,n_estimators= 1200,subsample=1,
                                               colsample_bytree=.7,
                                               objective='reg:squarederror',max_depth=6,eval_metric='rmse',
                                               reg_lambda=1.6681005372000592,reg_alpha=0.7196856730011514,
                                               tree_method='hist',n_jobs=-1,random_state=42))])




"""Have to do a manual stacking"""

"""reg_cv = RegressionCV1(n_splits=3)

xgb_oof = reg_cv.get_stacking_features(xgb_pipeline, X_train, y_train)
cat_oof = reg_cv.get_stacking_features(catboost, X_train, y_train)

X_meta_train = np.column_stack((xgb_oof, cat_oof))

meta_model = Ridge(alpha=1.0)
meta_model.fit(X_meta_train, y_train)

xgb_pipeline.fit(X_train, y_train)
catboost.fit(X_train, y_train)

xgb_test_preds = xgb_pipeline.predict(test_set_1)
cat_test_preds = catboost.predict(test_set_1)

X_meta_test = np.column_stack((xgb_test_preds, cat_test_preds))

final_prediction = meta_model.predict(X_meta_test)


submissions = pd.DataFrame({
    'id':test_set['id'],
    'exam_score': final_prediction
})

submissions.to_csv('Submissions_csv/testscores_submission1.csv',index=False)"""