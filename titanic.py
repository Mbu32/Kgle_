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
y= train_set[['Survived']].copy()
y_series = train_set['Survived'].copy()

#print(train_set.info())
#print(train_set.describe())
#print(train_set.Cabin)




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



"""name_features = ['Title_val', 'len_of_name', 'total_name', 'Title_len_interaction']
corr_matrix = X[name_features].corr()
print("Correlation between name features:")
print(corr_matrix)"""
X = X.drop(['len_of_name','total_name','Title_len_interaction'],axis=1) #highly correlated and Title_val was contributing the most at 55% correlation with survival






# every feature has 891 except age (714 data points) and cabin(204 data points). lets look into those
#Age, we should also bin
imputing = SimpleImputer(strategy='median')
X[['Age']]= imputing.fit_transform(X[["Age"]])

X.Age.hist()
plt.close()
age_bins = [0, 12, 18, 35, 60, np.inf]
X['age_group']= pd.cut(X['Age'],age_bins,labels=[1,2,3,4,5])
X['age_group']=X['age_group'].astype(float)






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





#what features we havent done yet: Fare,Parch,PassengerID,Embarked!
print(X.Embarked.head(25))





correlation_added_feature = X.corrwith(y_series,numeric_only=True)
print(correlation_added_feature.sort_values(ascending=False))
"""
Title_val                0.549622
Title_len_interaction    0.478919
total_name               0.374487
len_of_name              0.332350
Cabin_Val                0.301116
Fare                     0.257307
Parch                    0.081629
PassengerId             -0.005007
SibSp                   -0.035322
Age                     -0.064910
Pclass                  -0.338481


Title_val definitely giving us the best predictor thus far ~ and beats our interaction term as well as length of name
alone

update feature list
"""


features = ['PassengerId', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch',
       'Ticket', 'Fare', 'Cabin', 'Embarked', 'len_of_name', 'Title',
       'Title_val', 'total_name', 'Title_len_interaction', 'age_group',
       'Cabin_Val']

X.hist()
plt.close()

