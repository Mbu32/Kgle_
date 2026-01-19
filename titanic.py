import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsRegressor
from pathlib import Path
import tarfile
import urllib.request
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import confusion_matrix, precision_score,recall_score, f1_score, accuracy_score, precision_recall_curve
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, GridSearchCV, RandomizedSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.compose import TransformedTargetRegressor, ColumnTransformer
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.metrics import root_mean_squared_error
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.svm import LinearSVC, SVC



test_set = pd.read_csv('data_kaggle/test.csv')
train_set = pd.read_csv('data_kaggle/train.csv')


features = ['PassengerId', 'Pclass', 'Name', 'Sex', 'Age', 
            'SibSp','Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']
X = train_set[features].copy()
y= train_set[['Survived']].copy()
y_series = train_set['Survived'].copy()

#print(train_set.info())
#print(train_set.describe())
#print(train_set.Cabin)





'''Encoding Sex & Embarked feature''' 
emb_imp = SimpleImputer(strategy='most_frequent')
X[['Embarked']]=emb_imp.fit_transform(X[['Embarked']])



cat_features = ['Sex','Embarked']
cat_encoder = OneHotEncoder(sparse_output=False)
encoded_sex = cat_encoder.fit_transform(X[cat_features])
feature_names = cat_encoder.get_feature_names_out(cat_features)
cat_df = pd.DataFrame(encoded_sex,columns=feature_names,index=X.index)
X= pd.concat([X,cat_df],axis=1)










'''NAME feature'''
#print(train_set.Name.head(20)) #Some names have Master, some named do not... Indicator? Could length of name also tell us about status?, Lets see...
X['len_of_name'] = X['Name'].str.len()

X['Title']=X['Name'].str.extract('([A-Za-z]+)\.',expand=False)
title_map = {
    'Mrs': 4,      # Married women (had the highest survival)
    'Miss': 3,     # Unmarried women (also had high survival)
    'Master': 2,   # Boys (not as high but still decent odds)
    'Dr': 2,       # Professionals (decebt)
    'Rev': 1,      # Clergy (mixed)
    'Mr': 1,       # Adult men (had thelowest survival)
    'Col': 2,      # Military
    'Mlle': 3,     # Mademoiselle (French Miss)
    'Mme': 4,      # Madame (French Mrs)
    'Ms': 3,       # Modern equivalent
    'Don': 1,      # Spanish title
    'Dona': 4      # Spanish Mrs
}
X['Title_val']=X['Title'].map(title_map).fillna(0)

X['total_name'] = X['Title_val']+X['len_of_name']

#try interaction term?
X['Title_len_interaction'] = X['Title_val'] * X['len_of_name']


"""
Checking how correlated the new features are.

name_features = ['Title_val', 'len_of_name', 'total_name', 'Title_len_interaction']
corr_matrix = X[name_features].corr()
print("Correlation between name features:")
print(corr_matrix)"""
X = X.drop(['len_of_name','total_name','Title_len_interaction'],axis=1) #highly correlated and Title_val was contributing the most at 55% correlation with survival












'''AGE, scaled and binned. Ill try both see what works best?'''
# every feature has 891 except age (714 data points) and cabin(204 data points). lets look into those
#Age, we should also bin
imputing = SimpleImputer(strategy='median')
X[['Age']]= imputing.fit_transform(X[["Age"]])

age_scaler = StandardScaler()
X['Age_scaled']=age_scaler.fit_transform(X[['Age']])
age_bins = [0, 12, 18, 35, 60, np.inf]
X['age_group']= pd.cut(X['Age'],age_bins,labels=[1,2,3,4,5])
X['age_group']=X['age_group'].astype(float)








'''CABIN'''
#Cabin, Most people I'm assuming did not have a cabin, so we can replace NaN with 0 to indicate no Cabin.
# and 1 with cabin? lets try that and see what correlation we get. but theres levels to  cabins...
#lets try to stick with A-5,B-4,C-3,D-2,E-1, NaN-0
'''
Maybe A -9, B - 8, C - 7, D-6, E-5,F-4,G-3
'''
deck_mapping = {
    'A': 8, 'B': 7, 'C': 6, 'D': 5, 'E': 4,
    'F': 3, 'G': 2, 'T': 1
}
X['Cabin_Val']=X.Cabin.str[0].map(deck_mapping).fillna(0)
#Cabin FIxed ~ 









'''Fare feature combined with Parch and Sibsp'''

X['Fare_Parch_sibsp'] = X['Fare']/(X['Parch']+X['SibSp']+1)



'''Ticket Feature'''
X = X.drop("Ticket",axis=1)








'''Correlation of features with Survival'''
correlation_added_feature = X.corrwith(y_series,numeric_only=True)
#print(correlation_added_feature.sort_values(ascending=False))
"""
Title_val                  0.549622
Sex_female                 0.543351
Cabin_Val                  0.301116
Fare                       0.257307
Fare_Parch_sibsp           0.221600
Fare_Parch_sibsp_scaled    0.221600
Embarked_C                 0.168240
Parch                      0.081629
Embarked_Q                 0.003650
PassengerId               -0.005007
SibSp                     -0.035322
Age                       -0.064910
Age_scaled                -0.064910
age_group                 -0.093191
Embarked_S                -0.149683
Pclass                    -0.338481
Sex_male                  -0.543351

Title_val definitely giving us the best predictor thus far ~ and beats our interaction term as well as length of name
alone

update feature list
"""





#Took out Sex,Name,Age,SibSp,Fare, Parch, Title, Cabin,Embarked  will need to decide between Age Scaled/Group, Take out Embarked, 
updated_features = ['PassengerId', 'Pclass', 'Sex_female', 'Sex_male',
       'Embarked_C', 'Embarked_Q', 'Embarked_S',  'Title_val',
       'Age_scaled','age_group' , 'Cabin_Val', 'Fare_Parch_sibsp',
       'Fare_Parch_sibsp_scaled']

updated_features_ = ['PassengerId', 'Pclass', 'Sex_female', 'Sex_male',
       'Embarked_C', 'Embarked_Q', 'Embarked_S',  'Title_val',
       'age_group', 'Cabin_Val', 'Fare_Parch_sibsp']

"""
print(X[updated_features_].head(25))

vif = pd.DataFrame()
vif['feature']=X[updated_features_].columns
vif['VIF']=[variance_inflation_factor(X.values,i)
            for i in range(X.shape[1])]
print(vif)"""

'''Log Regression'''
#model_log = LogisticRegression(max_iter=10000)
#y_pred = cross_val_predict(model_log,X[updated_features_],y_series,cv=3,method='predict')





'''SGDClassifier'''
sgd_reg = SGDClassifier(loss='log_loss',
                        max_iter=3000,
                        tol=1e-5,
                        penalty='l1',
                        alpha=0.006,
                        eta0=0.01,
                        shuffle=True,
                        n_iter_no_change=100,
                        random_state=34,
                        verbose=0,
                        learning_rate='adaptive')


"""param_grid = [
    {'alpha': [.001,.006,.0001]},
    {'learning_rate':['adaptive','invscaling']},
    {'penalty':['l2','elasticnet']}
  
]
grid_search = GridSearchCV(sgd_reg,param_grid,cv=3,scoring='neg_root_mean_squared_error')
grid_search.fit(X[updated_features],y_series)
print(grid_search.best_estimator_)
"""
#y_pred_sgd = cross_val_predict(sgd_reg,X[updated_features_],y_series,cv=3,method='predict')



#acc_score = accuracy_score(y,y_pred_sgd)
#cm = confusion_matrix(y,y_pred_sgd)
#print(cm,acc_score)



#acc_score_log = accuracy_score(y,y_pred)
#cm_log = confusion_matrix(y,y_pred)
#print(cm_log,acc_score_log)





'''
SGD CLASSIFIER:
[[487  62]
 [127 215]] 0.7878787878787878

 

Logistic Regression
[[461  88]
 [ 97 245]] 0.792368125701459
'''



updated_features_svc = ['Pclass', 'Sex_female', 'Sex_male',
       'Embarked_C', 'Embarked_Q', 'Embarked_S',  'Title_val',
       'Age', 'Cabin_Val', 'Fare_Parch_sibsp']

'''SVC'''



"""
lin_pipeline = Pipeline([('scaler',StandardScaler()),
                        ('svc',SVC(kernel='rbf',C=1,gamma='scale',random_state=42))] )

param_dist2 = {'svc__C':np.logspace(-3,2,20),
               'svc__gamma':['scale', 'auto'] + list(np.logspace(-4, 1, 20)),
               'svc__class_weight': [None],
              'svc__tol':[1e-3]}


randoms = RandomizedSearchCV(lin_pipeline,param_distributions=param_dist2,
                             n_iter=30,
                             cv=5,
                             scoring='accuracy',
                             random_state=42,
                             n_jobs= 1
                             )
randoms.fit(X[updated_features_svc],y_series)
print('\n',randoms.best_params_,'\n',randoms.best_score_)"""

'''
Linear SVR
 {'svc__tol': 5e-06, 'svc__class_weight': None, 'svc__C': np.float64(10.0)} 
 0.792348251836043

 

Poly SVR

 {'svc__tol': 0.001, 'svc__gamma': np.float64(0.01), 'svc__degree': 3, 'svc__coef0': 1.5, 'svc__C': np.float64(16.666666666666668)}
 0.8226727763480008


rbf SVR

 {'svc__tol': 0.001, 'svc__gamma': np.float64(0.07847599703514607), 'svc__class_weight': None, 'svc__C': np.float64(2.636650898730358)}
 0.8238089259933463

'''

######PIPELINE FOR RBF SVR



def feature_ratio(X):
    return ((X[:,0]/((X[:,1]+X[:,2])+1)).reshape(-1,1))

def ratio_name(function_transformer, feature_names_in):
    return['Fare_Parch_sibsp']

def ratio_pipeline():
    return(make_pipeline(
        SimpleImputer(strategy='most_frequent'),
        FunctionTransformer(feature_ratio,feature_names_out=ratio_name),
        StandardScaler()
    ))


def name(X):
    title_map = {
    'Mrs': 4,'Miss': 3,'Master': 2,'Dr': 2,'Rev': 1,'Mr': 1,       
    'Col': 2,'Mlle': 3,'Mme': 4,'Ms': 3, 'Don': 1,'Dona': 4}
    titles=(X['Name'].str.extract('([A-Za-z]+)\.',expand=False)).map(title_map).fillna(0)
    return(titles.values.reshape(-1,1))


def cabin_val(X):
    deck_mapping = {
    'A': 8, 'B': 7, 'C': 6, 'D': 5, 'E': 4,
    'F': 3, 'G': 2, 'T': 1}

    cabin_values = X.Cabin.str[0].map(deck_mapping).fillna(0)
    return(cabin_values.values.reshape(-1,1))


cat_pipeline = make_pipeline(
    SimpleImputer(strategy='most_frequent'),
    OneHotEncoder(handle_unknown='ignore')
)


name_pipeline = make_pipeline(
    FunctionTransformer(name,validate=False),
    StandardScaler())


cabin_pipeline=make_pipeline(
    FunctionTransformer(cabin_val,validate=False),
    StandardScaler()
)

default_num_pipeline = make_pipeline(SimpleImputer(strategy='median'),
                                     StandardScaler())



preprocessing = ColumnTransformer([
    ('Fare_Parch_sibsp', ratio_pipeline(),['Fare','Parch','SibSp']),
    ('map_name',name_pipeline,['Name']),
    ('map_cabin',cabin_pipeline,['Cabin']),
    ('cat_feat',cat_pipeline,['Sex','Embarked']),
],remainder=default_num_pipeline)


full_pipeline = Pipeline([
    ('preprocessing', preprocessing),
    ('classifier', SVC(tol=.001,gamma=.07847599703514607,C=2.636650898730358,
                       class_weight=None,random_state=42))
])

 

features = ['Name', 'Cabin', 'Sex', 'Embarked', 'Pclass', 
            'Age', 'SibSp', 'Parch', 'Fare']

test_set = pd.read_csv('data_kaggle/test.csv')
train_set = pd.read_csv('data_kaggle/train.csv')

X = train_set[features].copy()
y_series = train_set['Survived'].copy()


test_set_f = test_set[features].copy()


full_pipeline.fit(X,y_series)
predictions = full_pipeline.predict(test_set_f)


submission = pd.DataFrame({
    'PassengerId': test_set['PassengerId'],
    'Survived':predictions
})


submission.to_csv('titanic_submission.csv',index=False)