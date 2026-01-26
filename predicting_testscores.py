import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
import tarfile
import os
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler, FunctionTransformer, PolynomialFeatures, MinMaxScaler,RobustScaler
from sklearn.compose import TransformedTargetRegressor, ColumnTransformer, make_column_selector, make_column_transformer
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics import root_mean_squared_error
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.model_selection import cross_val_score, cross_val_predict, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.svm import LinearSVR, SVR
from sklearn.ensemble import VotingRegressor, StackingRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression, Ridge,Lasso,ElasticNet
from sklearn.svm import LinearSVR, SVR
from sklearn.inspection import permutation_importance
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from perm_class import RegressionCV, ClusterSimilarity
from sklearn.preprocessing import PowerTransformer
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.random_projection import GaussianRandomProjection
from sklearn.decomposition import PCA


print("run",'\n')
#most likely, I have a feeling the best way to calculate this would be through a Decision tree method. 
#could even be a regression tbh
train = pd.read_csv('data_playground/train.csv')
test_set = pd.read_csv('data_playground/test.csv')



#print(train.info())

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


X = train[features].copy()
y = train[label].copy()
ids = train['id'].copy()

X_train, X_val , y_train, y_val = train_test_split(X,y,test_size=.2,random_state=42)



'''FEATURE ENGINEERING'''
#Interactions terms = Sleep_hours x Sleep_quality ... study_hours x internet access .... class attendance x course

"""sleep_q_map = {'poor':1,'average':2,'good':3}
X_train['sleep_q_encoded'] = X_train['sleep_quality'].map(sleep_q_map)

X_train['sleep_qualityxhours'] = X_train['sleep_q_encoded']*X_train['sleep_hours']


# 2
internet_access_map = {'yes':2,"no":1}
X_train['internet_encoded'] = X_train['internet_access'].map(internet_access_map)

X_train['studhours_internet'] = X_train['internet_encoded']*X_train['study_hours']"""

#3 Going to leave this out for now. third interaction term I mean.











#Pipelines for trees

"""
def internet_interaction(X):
    study_hours = X[:,1].astype(float)
    internet_encoded = np.where(X[:,0]=='yes' ,2,1)
    return((internet_encoded*study_hours).reshape(-1,1))

def sleep_interaction(X):
    sleep_q_map = {'poor':0,'average':1,'good':2}
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
        ],remainder='drop', verbose_feature_names_out=True)"""



#preprocessing tree I'll use for RandomForest, XGboost, LightGBM, 
#for Catboost I'll have to switch it up because we dont need onehotencoder for that one
"""rf_pipeline = Pipeline([
    ('preprocessing',preprocessing_tree),
    ('model_rf', RandomForestRegressor(random_state=42))
]
)
light_pipeline = Pipeline([
    ('preprocessing',preprocessing_tree),
    ('model_lg', LGBMRegressor(random_state=42,verbose=-1))
]
)"""
#inital check on Tree Models CV RMSE:
'''
cv_evaluator = RegressionCV(n_splits=3,shuffle=True,random_state=42)
rf_scores = cv_evaluator.evaluate(model=rf_pipeline,X=X_train,y=y_train)
light_scores = cv_evaluator.evaluate(model=light_pipeline,X=X_train,y=y_train)
cat_scores = cv_evaluator.evaluate(model=catboost,X=X_train,y=y_train,is_catboost=True)

print(light_scores,'\n',cat_scores)




Randomforest CV RMSE: 9.726 ± 0.018
Lightgbm: {'rmse_mean': 9.447802217749887, 'rmse_std': np.float64(0.014736158789610541), 'rmse_per_fold': [9.467744016819557, 9.44307298314212, 9.432589653287987]} 
CatBoost: {'rmse_mean': 8.769734450398255, 'rmse_std': np.float64(0.014535704008107124), 'rmse_per_fold': [8.789620570202096, 8.764301015300843, 8.755281765691828]}

'''



#CatBoost Randomized_search
"""
catboost = CatBoostRegressor(random_state=42,verbose=0,loss_function='RMSE',eval_metric='RMSE',cat_features=['gender','course','exam_difficulty',
                                                          'study_method','facility_rating',
                                                          'sleep_quality','internet_access'])



mapping_catboost = {
    'iterations': [100, 200, 300, 500],
    'learning_rate': [0.01, 0.03, 0.05, 0.1, 0.2],
    'depth': [4, 6, 8, 10],
    'l2_leaf_reg': [1, 3, 5, 7, 9],
    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
    'colsample_bylevel': [0.6, 0.7, 0.8, 0.9, 1.0],
    'min_data_in_leaf': [1, 3, 5, 10, 20]
}


search = catboost.randomized_search(param_distributions=mapping_catboost,X=X_train,y=y_train,cv=3,
                                    n_iter=15,
                                    partition_random_seed=42,
                                    calc_cv_statistics=True,
                                    search_by_train_test_split=True,
                                    refit=True,shuffle=True,stratified=None,
                                    train_size=.8,
                                    verbose=True    
)


for param in mapping_catboost.keys():
    print(f"{param}: {catboost.get_param(param)}")

best_params = search['params']

print("Best params:", best_params)
"""






#RandomSearchCV for XGBoost

"""mapping_xgb = {'model__learning_rate':np.logspace(-2,-.5,10),
               'model__max_depth':[2,3,4,5,6],
               'model__n_estimators':[300,500,800,1200],
               'model__subsample':np.linspace(.6,1,5),
               'model__colsample_bytree':np.linspace(.6,1,5),
               'model__reg_lambda':np.logspace(-2,2,10),
               'model__reg_alpha':np.logspace(-3,1,8)}



search = RandomizedSearchCV(estimator=full_tree_pipeline,
                            param_distributions=mapping_xgb,
                            n_iter=25,
                            cv=3,scoring='neg_root_mean_squared_error',
                            verbose=1,
                            random_state=42,
                            error_score='raise'
                            )
search.fit(X_train,y_train)
best_model = search.best_params_
score_tree = search.score(X_val,y_val)
print(best_model,'\n',score_tree)"""



#Model after RandomSearchCV parameters:
"""
Xgb_pipeline = Pipeline(
    [('preprocessing',preprocessing_tree),
    ('model',XGBRegressor(learning_rate=0.041700904432450864,n_estimators= 1400,subsample=0.9280934509687381,
                               colsample_bytree=0.6825330330763264,
                               objective='reg:squarederror',
                               max_depth=6,eval_metric='rmse',
                               reg_lambda=1.3866486140614158,reg_alpha=0.8880652023044383,
                               tree_method='hist',n_jobs=-1,
                               random_state=42))]
)
"""

#Results of XGB:
'''
{'model__subsample': np.float64(1.0),
 'model__reg_lambda': np.float64(1.6681005372000592),
 'model__reg_alpha': np.float64(0.7196856730011514), 
 'model__n_estimators': 1200, 
 'model__max_depth': 6, 
 'model__learning_rate': np.float64(0.046415888336127774), 
 'model__colsample_bytree': np.float64(0.7)}
 
 score: -9.383293361587777

'''









#second part.

# LINEAR MODELS PLUS THEIR PIPELINE + Distribution Check, anything else? Well who knows my friend, who knows

#starting with distribution check



'''Pipelines for linear models'''

num_poly_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('poly', PolynomialFeatures(
        degree=2,
        interaction_only=True,
        include_bias=False
    )),
    #('scaler', StandardScaler())
])

cat_pipeline = make_pipeline(
    SimpleImputer(strategy='most_frequent'),
    OneHotEncoder(handle_unknown='ignore'),
    #StandardScaler(with_mean=False)
)


preprocessing_linear_models = ColumnTransformer([
    ('num_poly', num_poly_pipeline,
     ['study_hours', 'sleep_hours', 'class_attendance', 'age']),
    ('cat', cat_pipeline,
     ['internet_access','gender','course','exam_difficulty',
      'study_method','facility_rating','sleep_quality']),
], remainder='drop')





kn_pipeline = Pipeline([('preprocessing',preprocessing_linear_models),
                        ('scaler',StandardScaler()),
                        ('PCA',PCA(n_components=10)),
                        ('model',KNeighborsRegressor())])



param_distributions = {
    'model__n_neighbors': np.arange(10,50,10),  
    'model__weights': ['uniform', 'distance'],
    'model__p': [1, 2],
    'model__leaf_size': np.arange(20, 50, 10),  
    'model__metric': ['minkowski', 'manhattan', 'euclidean', 'chebyshev']
}



search = RandomizedSearchCV(
    kn_pipeline,
    param_distributions=param_distributions,
    n_iter=15,  
    cv=3,
    scoring='neg_root_mean_squared_error',  
    n_jobs=-1,
    verbose=1,
    random_state=42
)

search.fit(X_train,y_train)
print(f"\nBest RMSE: {-search.best_score_:.4f}")
print(f"Best parameters: {search.best_params_}")




#kn_pipeline = Pipeline([('preprocessing',preprocessing_linear_models),('model',KNeighborsRegressor())])
#ridge_pipeline = Pipeline([('preprocessing', preprocessing_linear_models),('model', Ridge(alpha=1.0))])
#lasso_pipeline = Pipeline([('preprocessing', preprocessing_linear_models),('model', Lasso(alpha=0.01, max_iter=5000))])
#elastic_pipeline = Pipeline([('preprocessing', preprocessing_linear_models),('model', ElasticNet(alpha=0.01, max_iter=5000))])


#reg_cv = RegressionCV(n_splits=3,shuffle=True,random_state=42)


#ridge_score = reg_cv.evaluate(model=ridge_pipeline,X=X_train,y=y_train)
#lasso_score = reg_cv.evaluate(model=lasso_pipeline,X=X_train,y=y_train)
#kn_score = reg_cv.evaluate(model=kn_pipeline,X=X_train,y=y_train)
#elas_score = reg_cv.evaluate(model=elastic_pipeline,X=X_train,y=y_train)


















'''
{'rmse_mean': np.float64(10.81981755504166), 'rmse_std': np.float64(0.008024932325262916), 'rmse_per_fold': [10.830917592933842, 10.812220299584643, 10.816314772606495]} 
 {'rmse_mean': np.float64(9.61703441587507), 'rmse_std': np.float64(0.004846259489596193), 'rmse_per_fold': [9.619038832675935, 9.610356286269488, 9.621708128679785]}    
 {'rmse_mean': np.float64(9.617034370511687), 'rmse_std': np.float64(0.0048451305058780985), 'rmse_per_fold': [9.61903929616415, 9.610357567259065, 9.621706248111844]} 
 {'rmse_mean': np.float64(9.61832344826526), 'rmse_std': np.float64(0.004779318574393633), 'rmse_per_fold': [9.620992909965025, 9.611611142708465, 9.622366292122292]}  

'''


"""num_features = [
    'num_poly__study_hours',
    'num_poly__sleep_hours',
    'num_poly__class_attendance',
    'num_poly__age'
]

for col in num_features:
    sns.kdeplot(X_df[col])
    plt.title(col)
    plt.show()
"""








"""
kn_scores = cross_val_score(kn_pipeline, X_train, y_train, 
                           cv=3, scoring='r2', n_jobs=-1)

print(f"KNN CV R² scores: {kn_scores}")
print(f"Mean R²: {kn_scores.mean():.3f} (±{kn_scores.std():.3f})")

param_grid_knn = {
    'knn__n_neighbors': [3, 5, 7, 10, 15],
    'knn__weights': ['uniform', 'distance'],
    'knn__metric': ['euclidean', 'manhattan', 'minkowski'],
    'knn__p': [1, 2],  
}
"""
