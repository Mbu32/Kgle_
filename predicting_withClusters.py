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
from sklearn.metrics import root_mean_squared_error, silhouette_score
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
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import optuna 
from optuna.integration import CatBoostPruningCallback


start_time = time.time()


train = pd.read_csv('data_playground/train.csv')
test_set = pd.read_csv('data_playground/test.csv')

features = [ 'age', 'gender', 'course', 'study_hours', 'class_attendance',
       'internet_access', 'sleep_hours', 'sleep_quality', 'study_method', 
       'facility_rating', 'exam_difficulty']
label = 'exam_score'

X = train[features].copy()
y = train[label].copy()
test_set_1 = test_set[features].copy()

#high study flag and study bin num


num_features = ['study_hours', 'class_attendance', 'sleep_hours', 'age']
cat_features = [ 'gender', 'course', 
       'internet_access', 'sleep_quality', 'study_method', 
       'facility_rating', 'exam_difficulty']





"""st = StandardScaler()
X_num_scaled=st.fit_transform(X[num_features])
kmeans = KMeans(n_init=10,n_clusters=4,random_state=42)
all_clusters = kmeans.fit_predict(X_num_scaled)
X['student_cluster'] = all_clusters"""





X_train, X_val , y_train, y_val = train_test_split(X,y,test_size=.2,random_state=42)

"""test_num_scaled = st.transform(test_set_1[num_features])
test_set_1['student_cluster'] = kmeans.predict(test_num_scaled)"""










"""def create_student_archetypes(df):
    
    df = df.copy()
    
    df['perfect_student'] = (
        (df['class_attendance'] >= 90) &
        (df['study_hours'] >= 6) & 
        (df['sleep_hours'].between(7, 9)) &
        (df['sleep_quality'].isin(['Good', 'average'])) &
        (df['facility_rating'].isin(['high', 'medium']))
    ).astype(int)
    
    df['burnout_risk'] = (
        (df['study_hours'] >= 8) &
        (df['sleep_hours'] <= 6) 
    ).astype(int)
    
    df['social_butterfly'] = (
        (df['class_attendance'] >= 85) &
        (df['study_hours'] <= 4)
    ).astype(int)


    df['balanced_achiever'] = (
        (df['study_hours'].between(5, 7)) &
        (df['class_attendance'] >= 80) &
        (df['sleep_hours'].between(7, 8))
    ).astype(int)

    return df


X_train = create_student_archetypes(X_train)
X_val = create_student_archetypes(X_val)
test_set_1 = create_student_archetypes(test_set_1)

archetype_cols = ['perfect_student', 'burnout_risk', 'social_butterfly', 
                  'balanced_achiever']"""










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
           #'perfect_student', 'burnout_risk', 'social_butterfly', 'balanced_achiever'] #'student_cluster',


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
















'''Optimizing Ridge'''
"""def objective_ridge(trial):

    params = {
        'alpha': trial.suggest_float('alpha', 1e-4, 100.0, log=True),
        'solver': trial.suggest_categorical('solver', ['svd', 'cholesky', 'lsqr', 'sparse_cg']),
        'max_iter': trial.suggest_int('max_iter', 1000, 10000),
        'tol': trial.suggest_float('tol', 1e-6, 1e-3, log=True),
        'fit_intercept': trial.suggest_categorical('fit_intercept', [True, False]),
    }
    
    model = Pipeline([
        ('preprocessing', preprocessing_numeric),
        ('scaler', StandardScaler()),
        ('ridge', Ridge(**params, random_state=42))
    ])

    scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=3,
        scoring='neg_root_mean_squared_error',
        n_jobs=-1
    )

    return -scores.mean()

study_ridge = optuna.create_study(direction="minimize")
study_ridge.optimize(objective_ridge, n_trials=20)

print("Best RMSE:", study_ridge.best_value)
print("Best params:")
for k, v in study_ridge.best_params.items():
    print(f"  {k}: {v}")
"""









'''
Trial 1 finished with value: 8.767517031291252 and parameters: {'depth': 6, 'learning_rate': 0.19187636827736218, 'l2_leaf_reg': 8.287111010615796, 'subsample': 0.6862124700100836, 'min_data_in_leaf': 30}. Best is trial 1 
with value: 


'''


"""CATBOOST OPTIMIZATION OTUNA"""
"""def objective_cat(trial):

    params = {
        "depth": trial.suggest_int("depth", 6, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.3, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf',10,40),
        "iterations": 800,
        "loss_function": "RMSE",
        "eval_metric": "RMSE",
        "random_seed": 42,
        "verbose": 0,
        "early_stopping_rounds": 50
    }

    model = CatBoostRegressor(**params,cat_features=cat_cols)


    cv = RegressionCV1(n_splits=3)
    
    # We do NOT pass eval_set here. 
    # To speed up, we rely on lower depth and fewer folds.
    result = cv.evaluate(
        model,
        X_train,
        y_train
    )
    )

    return result['rmse_mean']

study_cat = optuna.create_study(direction="minimize")
study_cat.optimize(objective_cat, n_trials=30)

print("Best RMSE:", study_cat.best_value)
print("Best params:")
for k, v in study_cat.best_params.items():
    print(f"  {k}: {v}")"""



catboost = CatBoostRegressor(random_state=42,
                             verbose=0,loss_function='RMSE',
                             eval_metric='RMSE',
                             cat_features=['gender','course','exam_difficulty',
                                                          'study_method','facility_rating',
                                                          'sleep_quality','internet_access'], #student_cluster
                            iterations=300,
                            subsample= 0.6862124700100836,
                            learning_rate= 0.19187636827736218,
                            depth=6,
                            colsample_bylevel=.6,
                            l2_leaf_reg=8.287111010615796,
                            min_data_in_leaf=30)
#8.767517031291252


xgb_pipeline = Pipeline([('preprocessing',preprocessing_numeric),
                         ('model',XGBRegressor(learning_rate=0.020270696458787967,n_estimators= 1589,subsample=0.7011843457443048,
                                               colsample_bytree= 0.7508962426277108,
                                               objective='reg:squarederror',max_depth=7,eval_metric='rmse',
                                               reg_lambda= 1.3075704748577666,reg_alpha= 0.11818925588505813,
                                               tree_method='hist',n_jobs=-1,random_state=42))])
#8.750646188282467.



lin_pipeline = Pipeline([
    ('preprocessing',preprocessing_numeric),
    ('scaler',StandardScaler()),
    ('model',Ridge(tol=4.150176855285925e-06,solver='svd',max_iter=3191,alpha=0.0006788022834104169,fit_intercept=True))
])
#8.885







"""Have to do a manual stacking"""

reg_cv = RegressionCV1(n_splits=3)

lin_oof = reg_cv.get_stacking_features(lin_pipeline,X_train,y_train)
xgb_oof = reg_cv.get_stacking_features(xgb_pipeline, X_train, y_train)
cat_oof = reg_cv.get_stacking_features(catboost, X_train, y_train)

X_meta_train = np.column_stack((xgb_oof, cat_oof,lin_oof)) 

meta_model = Ridge(alpha=1.0)
meta_model.fit(X_meta_train, y_train)

lin_pipeline.fit(X_train,y_train)
xgb_pipeline.fit(X_train, y_train)
catboost.fit(X_train, y_train)

lin_test_preds = lin_pipeline.predict(test_set_1)
xgb_test_preds = xgb_pipeline.predict(test_set_1)
cat_test_preds = catboost.predict(test_set_1)

X_meta_test = np.column_stack((xgb_test_preds, cat_test_preds,lin_test_preds)) #
final_prediction = meta_model.predict(X_meta_test)


submissions = pd.DataFrame({
    'id':test_set['id'],
    'exam_score': final_prediction
})

submissions.to_csv('Submissions_csv/test_sub8_woclusters.csv',index=False)


