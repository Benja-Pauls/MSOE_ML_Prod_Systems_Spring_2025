import numpy as np
import pandas as pd
import dill
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
import os
import logging
from datetime import datetime

# Verify scikit-learn version
import sklearn
print(f"Using scikit-learn version: {sklearn.__version__}")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_csv_data(csv_path="/app/house_price_prediction_service.csv"):
    """Load real data from CSV file"""
    logger.info(f"Loading data from {csv_path}")
    
    try:
        # Read the CSV file
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(df)} rows from CSV file")
        
        # Add dummy values for missing enrichment data columns
        if 'population' not in df.columns:
            df['population'] = 20000
        
        if 'total_schools' not in df.columns:
            df['high_schools'] = 2
            df['middle_schools'] = 3
            df['primary_schools'] = 5
            df['other_schools'] = 1
            df['unknown_schools'] = 0
            df['total_schools'] = 11
        
        return df
    except Exception as e:
        logger.error(f"Error loading CSV data: {e}")
        raise

# Define feature engineering transformer
class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_new = X.copy()
        
        # Handle date features
        X_new['date'] = pd.to_datetime(X_new['date'])
        X_new['year'] = X_new['date'].dt.year
        X_new['month'] = X_new['date'].dt.month
        X_new['day_of_week'] = X_new['date'].dt.dayofweek
        X_new['day_of_year'] = X_new['date'].dt.dayofyear
        
        # Handle missing values in sqft_lot15
        X_new['sqft_lot15'] = X_new['sqft_lot15'].fillna(X_new['sqft_lot'])
        
        # Create derived features
        X_new['age'] = X_new['year'] - X_new['yr_built']
        X_new['renovated'] = (X_new['yr_renovated'] > 0).astype(int)
        X_new['total_sqft'] = X_new['sqft_living'] + X_new['sqft_lot']
        
        # Safe division operations
        X_new['price_per_sqft'] = np.where(X_new['sqft_living'] > 0, X_new['price'] / X_new['sqft_living'], 0)
        X_new['bed_bath_ratio'] = np.where(X_new['bathrooms'] > 0, X_new['bedrooms'] / X_new['bathrooms'], 0)
        X_new['total_rooms'] = X_new['bedrooms'] + X_new['bathrooms']
        X_new['school_density'] = np.where(X_new['population'] > 0, X_new['total_schools'] / X_new['population'] * 10000, 0)
        
        # Log transformations
        X_new['log_price'] = np.log1p(X_new['price'])
        X_new['log_sqft_living'] = np.log1p(X_new['sqft_living'])
        X_new['log_sqft_lot'] = np.log1p(X_new['sqft_lot'])
        X_new['log_population'] = np.log1p(X_new['population'])
        
        # Categorical features
        X_new['age_category'] = pd.cut(X_new['age'], 
                                       bins=[0, 10, 20, 40, 80, 150], 
                                       labels=['New', 'Recent', 'Mid', 'Old', 'Very Old'])
        X_new['size_category'] = pd.cut(X_new['sqft_living'], 
                                        bins=[0, 1000, 2000, 3000, 5000, 15000], 
                                        labels=['Tiny', 'Small', 'Medium', 'Large', 'Mansion'])
        
        # Replace any potential infinities
        X_new = X_new.replace([np.inf, -np.inf], np.nan)
        
        return X_new

# Define feature selector
class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, drop_columns=None):
        self.drop_columns = drop_columns or []
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X.drop(columns=self.drop_columns, errors='ignore')

def build_model_pipeline(df):
    """Build the model pipeline"""
    logger.info("Building model pipeline...")
    
    # Define columns to drop
    non_features = ['id', 'date', 'price', 'log_price', 'price_per_sqft', 'age_category', 'size_category']
    
    # Apply feature engineering
    feature_eng = FeatureEngineer()
    df_engineered = feature_eng.transform(df)
    
    # Select features
    feature_selector = FeatureSelector(drop_columns=non_features)
    X_selected = feature_selector.transform(df_engineered)
    
    # Identify feature types
    numeric_features = X_selected.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X_selected.select_dtypes(include=['object', 'category']).columns.tolist()
    
    logger.info(f"Numeric features: {len(numeric_features)}")
    logger.info(f"Categorical features: {len(categorical_features)}")
    
    # Create preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), numeric_features),
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ]), categorical_features)
        ]
    )
    
    # Build the pipeline
    pipeline = Pipeline([
        ('feature_engineer', FeatureEngineer()),
        ('feature_selector', FeatureSelector(drop_columns=non_features)),
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(
            n_estimators=100,  # Reduced for faster training
            max_depth=20,
            min_samples_split=2,
            random_state=42
        ))
    ])
    
    return pipeline, df_engineered['log_price']

if __name__ == "__main__":
    # Download CSV file if needed
    if not os.path.exists("/app/house_price_prediction_service.csv"):
        logger.error("CSV file not found")
        exit(1)
    
    # Load data from CSV
    df = load_csv_data()
    
    # Build model pipeline
    pipeline, y = build_model_pipeline(df)
    
    # Train the model
    logger.info("Training model...")
    pipeline.fit(df, y)
    
    # Save the model
    logger.info("Saving model...")
    with open("/app/model.pkl", "wb") as f:
        dill.dump(pipeline, f)
    
    logger.info("Model saved to /app/model.pkl")