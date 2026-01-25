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
from sklearn.compose import TransformedTargetRegressor, ColumnTransformer, make_column_selector, make_column_transformer
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics import root_mean_squared_error
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.model_selection import cross_val_score, cross_val_predict, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.svm import LinearSVR, SVR
from sklearn.ensemble import VotingRegressor, StackingRegressor, RandomForestRegressor, ExtraTreesRegressor,BaggingRegressor
import xgboost as xgb
from xgboost.callback import EarlyStopping
from sklearn.linear_model import LinearRegression

#most likely, I have a feeling the best way to calculate this would be through a Decision tree method. Maybe Kneighbour?
#could even be a regression tbh
train = pd.read_csv('data_playground/train.csv')
test = pd.read_csv('data_playground/test.csv')




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

X_train, X_val , y_train, y_val = train_test_split(X,y,test_size=.2,random_state=42,stratify= y)



'''FEATURE ENGINEERING'''
#Interactions terms = Sleep_hours x Sleep_quality ... study_hours x internet access .... class attendance x course

sleep_q_map = {'poor':1,'average':2,'good':3}
X_train['sleep_q_encoded'] = X_train['sleep_quality'].map(sleep_q_map)

X_train['sleep_qualityxhours'] = X_train['sleep_q_encoded']*X_train['sleep_hours']


# 2
internet_access_map = {'yes':2,"no":1}
X_train['internet_encoded'] = X_train['internet_access'].map(internet_access_map)

X_train['studhours_internet'] = X_train['internet_encoded']*X_train['study_hours']

#3 Going to leave this out for now. third interaction term I mean.








#Let's onehot encode
cat_features = ['gender','course','exam_difficulty',
                'study_method','facility_rating']

int_features =['sleep_quality','sleep_hours','study_hours','internet_access','study_hours','study_method']

num_features = ['age'] #since rest in interaction term

features = [ 'age', 'gender', 'course', 'study_hours', 'class_attendance',
       'internet_access', 'sleep_hours', 'sleep_quality', 'study_method', 
       'facility_rating', 'exam_difficulty',
       'sleep_quality_hours_interaction','Internet_studyhours_int']





'''Pipelines'''


def internet_interaction(X):
    internet_access_map = {'yes':2,"no":1}
    internet_encoded =(X[:,0].map(internet_access_map))
    study_method = X[:,1]
    return((internet_encoded*study_method).reshape(-1,1))

def sleep_interaction(X):
    sleep_q_map = {'poor':1,'average':2,'good':3},
    sleep_q=X[:,0].map(sleep_q_map)
    sleep_hours = X[:,1]
    return(sleep_q*sleep_hours).reshape(-1,1)


internet_interaction_pipeline = make_pipeline(
    SimpleImputer(strategy='median'),
    FunctionTransformer(internet_interaction,validate=False)
)

sleep_interaction_pipeline = make_pipeline(
    SimpleImputer(strategy='median'),
    FunctionTransformer(sleep_interaction,validate=False)
)


cat_pipeline = make_pipeline(
    SimpleImputer(strategy='most_frequent'),
    OneHotEncoder(handle_unknown='ignore')
)


default_num_pipeline = make_pipeline(
    SimpleImputer(strategy='median'),
)

tree_pipeline = Pipeline([
    ('preprocessing',ColumnTransformer([
        ('Internetxstudy_hours',internet_interaction_pipeline,['internet_access','study_hours']),
        ('sleep_encoded',sleep_interaction_pipeline,['sleep_quality','sleep_hours']),
        ('cat_feat',cat_pipeline,['gender','course','exam_difficulty','study_method','facility_rating','class_attendance']),
        ('numeric',default_num_pipeline,['age'])
    ])),
    ('model',xgb.XGBRegressor(learning_rate=.1,
                              n_estimators=100,
                              subsample=.6,
                              colsample_bytree=.6,
                              objective='reg:squarederror',
                              max_depth=3,
                              eval_metric='rmse',
                              reg_lambda=1,
                              reg_alpha=0,
                              tree_method='hist',
                              n_jobs=-1,
                              random_state=42
    ))
])

#RandomSearchCV 

mapping_xgb = {'model__learning_rate':np.logspace(-2,-.5,10),
               'model__max_depth':[2,3,4,5,6],
               'model__n_estimators':[300,500,800,1200],
               'model__subsample':np.linspace(.6,1,5),
               'model__colsample_bytree':np.linspace(.6,1,5),
               'model__reg_lambda':np.logspace(-2,2,10),
               'model__reg_alpha':np.logspace(-3,1,8)}



search = RandomizedSearchCV(estimator=tree_pipeline,
                            param_distributions=mapping_xgb,
                            n_iter=25,
                            cv=3,scoring='neg_root_mean_squared_error',
                            verbose=1,
                            random_state=42)


search.fit(X_train,y_train)

best_model = search.best_params_
score_tree = search.score(X_val,y_val)
print(best_model)


#RandomForest





"""#Bagging -> VotingRegressor

bagged_lr = BaggingRegressor(estimator=RandomForestRegressor(),
                             n_estimators=11,
                             max_samples=.6,
                             random_state=42)




r_models = {
    'rf':RandomForestRegressor(random_state=42),
    'svr':SVR(random_state=42),
    'knn':KNeighborsRegressor(random_state=42),
    'xgb':xgb.XGBRegressor(random_state=42)
}


voting_regress = VotingRegressor(
    estimators=[r_models]

)"""