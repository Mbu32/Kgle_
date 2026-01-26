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


'''Pipelines'''


def internet_interaction(X):
    study_hours = X[:,1].astype(float)
    internet_encoded = np.where(X[:,0]=='yes' ,2,1)
    return((internet_encoded*study_hours).reshape(-1,1))

def sleep_interaction(X):
    sleep_q_map = {'poor':1,'average':2,'good':3}
    sleep_q= np.vectorize(sleep_q_map.get)(X[:,0])
    sleep_hours = X[:,1].astype(float)
    return(sleep_q*sleep_hours).reshape(-1,1)


def internet_hours_name(function_transformer,feature_names_in):
    return['Internetxstudy_hours']


def sleep_quality_name(function_transformer,feature_names_in):
    return['sleep_q_int']


def internet_interaction_pipeline():
    return(make_pipeline(
        SimpleImputer(strategy='most_frequent'),
        FunctionTransformer(internet_interaction,validate=False,
                            feature_names_out=internet_hours_name)))


def sleep_interaction_pipeline():
    return (make_pipeline(
        SimpleImputer(strategy='most_frequent'),
        FunctionTransformer(sleep_interaction,validate=False,
                            feature_names_out=sleep_quality_name)))


cat_pipeline = make_pipeline(
    SimpleImputer(strategy='most_frequent'),
    OneHotEncoder(handle_unknown='ignore')
)


default_num_pipeline = make_pipeline(
    SimpleImputer(strategy='median'),
)


preprocessing_tree = ColumnTransformer([
        ('Internetxstudy_hours',internet_interaction_pipeline(),['internet_access','study_hours']),
        ('sleep_q_int',sleep_interaction_pipeline(),['sleep_quality','sleep_hours']),
        ('cat_feat',cat_pipeline,['gender','course','exam_difficulty','study_method','facility_rating']),
        ('numeric',default_num_pipeline,['age','class_attendance'])
        ],remainder='drop', verbose_feature_names_out=True)




"""
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



catboost.fit(X,y)
prediction =catboost.predict(test_set_1)"""

Xgb_pipeline = Pipeline(
    [('preprocessing',preprocessing_tree),
    ('model',XGBRegressor(learning_rate=0.046415888336127774,n_estimators= 1200,subsample=1,
                               colsample_bytree=.7,
                               objective='reg:squarederror',
                               max_depth=6,eval_metric='rmse',
                               reg_lambda=1.6681005372000592,reg_alpha=0.7196856730011514,
                               tree_method='hist',n_jobs=-1,
                               random_state=42))]
)



mapping_xgb = {'model__learning_rate': loguniform(0.03, 0.07),  
               'model__max_depth': [5, 6, 7],
               'model__n_estimators': [1000, 1200, 1400],
               'model__subsample': uniform(0.9, 0.1),
               'model__colsample_bytree': uniform(0.65, 0.1),'model__reg_lambda': loguniform(1.0, 2.5),  'model__reg_alpha': loguniform(0.5, 1.0)  
}




search = RandomizedSearchCV(estimator=Xgb_pipeline,
                            param_distributions=mapping_xgb,
                            n_iter=15,
                            cv=3,scoring='neg_root_mean_squared_error',
                            verbose=1,
                            random_state=42,
                            error_score='raise'
                            )
search.fit(X_train,y_train)
best_model = search.best_params_
score_tree = search.score(X_val,y_val)
print(best_model,'\n',score_tree)






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