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
from xgboost import XGBRegressor
from xgboost.callback import EarlyStopping
from sklearn.linear_model import LinearRegression
from sklearn.inspection import permutation_importance

#most likely, I have a feeling the best way to calculate this would be through a Decision tree method. Maybe Kneighbour?
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


internet_interaction_pipeline = make_pipeline(
    SimpleImputer(strategy='most_frequent'),
    FunctionTransformer(internet_interaction,validate=False)
)

sleep_interaction_pipeline = make_pipeline(
    SimpleImputer(strategy='most_frequent'),
    FunctionTransformer(sleep_interaction,validate=False)
)


cat_pipeline = make_pipeline(
    SimpleImputer(strategy='most_frequent'),
    OneHotEncoder(handle_unknown='ignore')
)


default_num_pipeline = make_pipeline(
    SimpleImputer(strategy='median'),
)


preprocessing_tree = ColumnTransformer([
        ('Internetxstudy_hours',internet_interaction_pipeline,['internet_access','study_hours']),
        ('sleep_encoded',sleep_interaction_pipeline,['sleep_quality','sleep_hours']),
        ('cat_feat',cat_pipeline,['gender','course','exam_difficulty','study_method','facility_rating']),
        ('numeric',default_num_pipeline,['age','class_attendance'])
        ])

full_tree_pipeline = Pipeline([
    ('preprocessing',preprocessing_tree),
    ('model',XGBRegressor(learning_rate=0.046415888336127774,
                              n_estimators= 1200,
                              subsample=1,
                              colsample_bytree=.7,
                              objective='reg:squarederror',
                              max_depth=6,
                              eval_metric='rmse',
                              reg_lambda=1,
                              reg_alpha=0,
                              tree_method='hist',
                              n_jobs=-1,
                              random_state=42))
])

full_tree_pipeline.fit(X_train,y_train)

r=  permutation_importance(full_tree_pipeline,
                           X_val,
                           y_val,
                           n_repeats=10,
                           random_state=42,
                           n_jobs=-1)


preprocessor = full_tree_pipeline.named_steps['preprocessing']
feature_names = preprocessor.get_feature_names_out()
perm_fi = pd.DataFrame({
    'feature': feature_names,
    'importance': r.importances_mean
}).sort_values(by='importance', ascending=False)

print(perm_fi.head(15))



#RandomSearchCV 

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

"""
final_model = XGBRegressor(learning_rate=0.046415888336127774,n_estimators= 1200,subsample=1,
                               colsample_bytree=.7,
                               objective='reg:squarederror',
                               max_depth=6,eval_metric='rmse',
                               reg_lambda=1,reg_alpha=0,
                               tree_method='hist',n_jobs=-1,
                               random_state=42)


X_train_processed = preprocessing_tree.fit_transform(X_train)
X_val_processed = preprocessing_tree.transform(X_val)

final_model.fit(X_train_processed,y_train,
                       eval_set = [(X_val_processed,y_val)],
                       verbose=100)
"""


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








#subsmission
"""full_tree_pipeline.fit(X,y)
prediction = full_tree_pipeline.predict(test_set)

submissions = pd.DataFrame({
    'id':test_set['id'],
    'exam_score': prediction
})

submissions.to_csv('data_playground/testscores_submission.csv',index=False)"""