from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier 


X_test,y_test,X_train,y_train = [[5,5]]
predictions = (1,5)
# Define numeric and categorical columns
numeric_features = ['age', 'income', 'score']
categorical_features = ['gender', 'city', 'education']

# Preprocessor for models that need scaling
preprocessor_with_scaling = ColumnTransformer([
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])

# Preprocessor for tree models (no scaling)
preprocessor_no_scaling = ColumnTransformer([
    ('num', 'passthrough', numeric_features),  # No scaling!
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])

# Create model-specific pipelines
pipelines = {
    'xgboost': Pipeline([
        ('preprocessor', preprocessor_no_scaling),
        ('model', xgb.XGBClassifier())
    ]),
    
    'random_forest': Pipeline([
        ('preprocessor', preprocessor_no_scaling),
        ('model', RandomForestClassifier())
    ]),
    
    'logistic_regression': Pipeline([
        ('preprocessor', preprocessor_with_scaling),
        ('model', LogisticRegression())
    ]),
    
    'svm': Pipeline([
        ('preprocessor', preprocessor_with_scaling),
        ('model', SVC(probability=True))
    ]),
    
    'knn': Pipeline([
        ('preprocessor', preprocessor_with_scaling),
        ('model', KNeighborsClassifier())
    ])
}

# Use them in ensemble
for name, pipeline in pipelines.items():
    pipeline.fit(X_train, y_train)
    predictions[name] = pipeline.predict_proba(X_test)