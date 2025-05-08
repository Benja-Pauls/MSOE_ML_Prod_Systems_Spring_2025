import numpy as np
import pandas as pd
import logging
import pickle
import os
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define feature engineering transformer
class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom transformer for feature engineering on housing data.
    
    This transformer performs several operations:
    1. Extracts date components (year, month, day of week, day of year)
    2. Handles missing values in sqft_lot15
    3. Creates derived features like age, renovation status, total square footage
    4. Calculates price per sqft, bed/bath ratio, and other derived metrics
    5. Applies log transformations to normalize skewed distributions
    6. Creates categorical features from continuous variables
    """
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        # No fitting needed, just return self
        return self
    
    def transform(self, X):
        # Make a copy to avoid modifying the original
        X_new = X.copy()
        
        # Handle date features - extract components from the date column
        X_new['date'] = pd.to_datetime(X_new['date'])
        X_new['year'] = X_new['date'].dt.year
        X_new['month'] = X_new['date'].dt.month
        X_new['day_of_week'] = X_new['date'].dt.dayofweek
        X_new['day_of_year'] = X_new['date'].dt.dayofyear
        
        # Handle missing values in sqft_lot15
        X_new['sqft_lot15'] = X_new['sqft_lot15'].fillna(X_new['sqft_lot'])
        
        # Create derived features
        X_new['age'] = X_new['year'] - X_new['yr_built']  # Age of the house
        X_new['renovated'] = (X_new['yr_renovated'] > 0).astype(int)  # Binary renovation indicator
        X_new['total_sqft'] = X_new['sqft_living'] + X_new['sqft_lot']  # Total square footage
        
        # Safe division operations to handle zeros and edge cases
        X_new['price_per_sqft'] = np.where(X_new['sqft_living'] > 0, X_new['price'] / X_new['sqft_living'], 0)
        X_new['bed_bath_ratio'] = np.where(X_new['bathrooms'] > 0, X_new['bedrooms'] / X_new['bathrooms'], 0)
        X_new['total_rooms'] = X_new['bedrooms'] + X_new['bathrooms']
        X_new['school_density'] = np.where(X_new['population'] > 0, X_new['total_schools'] / X_new['population'] * 10000, 0)
        
        # Log transformations to handle skewed distributions
        X_new['log_price'] = np.log1p(X_new['price'])  # Target variable transformation
        X_new['log_sqft_living'] = np.log1p(X_new['sqft_living'])
        X_new['log_sqft_lot'] = np.log1p(X_new['sqft_lot'])
        X_new['log_population'] = np.log1p(X_new['population'])
        
        # Categorical features created from continuous variables
        # Age categories
        X_new['age_category'] = pd.cut(X_new['age'], 
                                      bins=[0, 10, 20, 40, 80, 150], 
                                      labels=['New', 'Recent', 'Mid', 'Old', 'Very Old'])
        # Size categories
        X_new['size_category'] = pd.cut(X_new['sqft_living'], 
                                        bins=[0, 1000, 2000, 3000, 5000, 15000], 
                                        labels=['Tiny', 'Small', 'Medium', 'Large', 'Mansion'])
        
        # Replace any potential infinities with NaN (will be handled by imputer later)
        X_new = X_new.replace([np.inf, -np.inf], np.nan)
        
        return X_new

# Define feature selector
class FeatureSelector(BaseEstimator, TransformerMixin):
    """
    Custom transformer to select features by dropping specified columns.
    
    This transformer removes columns that shouldn't be used as features,
    such as IDs, raw date fields, and derived columns that we don't want
    to use as direct inputs to the model.
    """
    def __init__(self, drop_columns=None):
        self.drop_columns = drop_columns or []
    
    def fit(self, X, y=None):
        # No fitting needed, just return self
        return self
    
    def transform(self, X):
        # Drop specified columns, ignoring any that don't exist
        return X.drop(columns=self.drop_columns, errors='ignore')

def build_model_pipeline():
    """
    Build the complete model pipeline for house price prediction.
    
    The pipeline consists of:
    1. Feature engineering - creates derived features from raw data
    2. Feature selection - removes unnecessary columns
    3. Preprocessing - handles missing values and scales numeric features
                     - one-hot encodes categorical features
    4. Regression model - RandomForest with optimized hyperparameters
    """
    logger.info("Building model pipeline...")
    
    # Define columns to drop during feature selection
    # These are columns we don't want to use directly as features
    non_features = ['id', 'date', 'price', 'log_price', 'price_per_sqft', 'age_category', 'size_category']
    
    # Create preprocessing pipeline with separate steps for numeric and categorical features
    preprocessor = ColumnTransformer(
        transformers=[
            # Numeric preprocessing: impute missing values with median, then scale
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), []),  # Will be filled in later
            # Categorical preprocessing: impute missing values with most frequent, then one-hot encode
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ]), [])  # Will be filled in later
        ]
    )
    
    # Build the final pipeline with all steps
    pipeline = Pipeline([
        # Step 1: Transform raw features into engineered features
        ('feature_engineer', FeatureEngineer()),
        # Step 2: Select which features to use (drop non-features)
        ('feature_selector', FeatureSelector(drop_columns=non_features)),
        # Step 3: Preprocess features (imputation, scaling, encoding)
        ('preprocessor', preprocessor),
        # Step 4: Train the regression model
        # Using hyperparameters tuned from notebook analysis
        ('regressor', RandomForestRegressor(
            n_estimators=200,  # Number of trees
            max_depth=30,      # Maximum depth of trees
            min_samples_split=2,  # Minimum samples required to split a node
            random_state=42    # For reproducibility
        ))
    ])
    
    return pipeline

def train_test_model(df):
    """
    Perform train-test split and train the model on the training data.
    Evaluate the model on the test data.
    
    Returns:
        tuple: (pipeline, metrics, features_used)
    """
    logger.info("Performing train-test split...")
    
    # Apply feature engineering to identify feature types
    feature_eng = FeatureEngineer()
    df_engineered = feature_eng.transform(df)
    
    # Define target variable
    y = df_engineered['log_price']
    
    # Define non-features to exclude
    non_features = ['id', 'date', 'price', 'log_price', 'price_per_sqft', 'age_category', 'size_category']
    
    # Create feature selector
    feature_selector = FeatureSelector(drop_columns=non_features)
    X_selected = feature_selector.transform(df_engineered)
    
    # Identify feature types for preprocessing
    numeric_features = X_selected.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X_selected.select_dtypes(include=['object', 'category']).columns.tolist()
    
    logger.info(f"Numeric features: {len(numeric_features)}")
    logger.info(f"Categorical features: {len(categorical_features)}")
    
    # Create train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        df, y, test_size=0.2, random_state=42
    )
    
    logger.info(f"Training set size: {len(X_train)}")
    logger.info(f"Test set size: {len(X_test)}")
    
    # Build and train the model pipeline
    pipeline = build_model_pipeline()
    
    # Update the preprocessor with the feature lists
    pipeline.named_steps['preprocessor'].transformers[0] = (
        'num', pipeline.named_steps['preprocessor'].transformers[0][1], numeric_features
    )
    pipeline.named_steps['preprocessor'].transformers[1] = (
        'cat', pipeline.named_steps['preprocessor'].transformers[1][1], categorical_features
    )
    
    # Train the model
    logger.info("Training model on training data...")
    pipeline.fit(X_train, y_train)
    
    # Evaluate on test set
    logger.info("Evaluating model on test data...")
    y_pred = pipeline.predict(X_test)
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((np.expm1(y_test) - np.expm1(y_pred)) / np.expm1(y_test))) * 100
    
    metrics = {
        'rmse': rmse,
        'r2': r2,
        'mape': mape
    }
    
    logger.info(f"Test RMSE: {rmse:.4f}")
    logger.info(f"Test R²: {r2:.4f}")
    logger.info(f"Test MAPE: {mape:.2f}%")
    
    # Return the trained pipeline, metrics, and features used
    return pipeline, metrics, numeric_features + categorical_features

def train_final_model(df):
    """
    Train the final model on all data.
    
    Returns:
        tuple: (pipeline, features_used)
    """
    logger.info("Training final model on all data...")
    
    # Apply feature engineering to identify feature types
    feature_eng = FeatureEngineer()
    df_engineered = feature_eng.transform(df)
    
    # Define target variable
    y = df_engineered['log_price']
    
    # Define non-features to exclude
    non_features = ['id', 'date', 'price', 'log_price', 'price_per_sqft', 'age_category', 'size_category']
    
    # Create feature selector
    feature_selector = FeatureSelector(drop_columns=non_features)
    X_selected = feature_selector.transform(df_engineered)
    
    # Identify feature types for preprocessing
    numeric_features = X_selected.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X_selected.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Build the model pipeline
    pipeline = build_model_pipeline()
    
    # Update the preprocessor with the feature lists
    pipeline.named_steps['preprocessor'].transformers[0] = (
        'num', pipeline.named_steps['preprocessor'].transformers[0][1], numeric_features
    )
    pipeline.named_steps['preprocessor'].transformers[1] = (
        'cat', pipeline.named_steps['preprocessor'].transformers[1][1], categorical_features
    )
    
    # Train the model on all data
    pipeline.fit(df, y)
    
    # Return the trained pipeline and features used
    return pipeline, numeric_features + categorical_features 