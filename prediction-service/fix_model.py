# fix_model.py
import numpy as np
import pandas as pd
import dill
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor

# Create a minimal model with scikit-learn 1.4.0
pipeline = Pipeline([
    ('preprocessor', StandardScaler()),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Train it on minimal data
X = np.array([[1, 2], [3, 4]])
y = np.array([1, 2])
pipeline.fit(X, y)

# Save it
with open("simple_model.pkl", "wb") as f:
    dill.dump(pipeline, f)

print("Created simple model with scikit-learn:", __import__('sklearn').__version__)