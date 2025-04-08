import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
import dill
import os
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_csv_data(csv_path="../notebook/house_price_prediction_service.csv"):
    """Load real data from CSV file"""
    logger.info(f"Loading data from {csv_path}")
    
    try:
        # Read the CSV file
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(df)} rows from CSV file")
        
        # Add dummy values for missing enrichment data columns
        # These would normally come from the PostgreSQL database
        if 'population' not in df.columns:
            df['population'] = 20000  # Placeholder value
            logger.info("Added dummy population column")
        
        if 'total_schools' not in df.columns:
            # Create placeholder school columns (high, middle, primary, other, unknown, total)
            df['high_schools'] = 2
            df['middle_schools'] = 3
            df['primary_schools'] = 5
            df['other_schools'] = 1
            df['unknown_schools'] = 0
            df['total_schools'] = 11
            logger.info("Added dummy school columns")
        
        return df
    except Exception as e:
        logger.error(f"Error loading CSV data: {e}")
        raise

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

def build_model_pipeline(df):
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
    
    # Apply feature engineering to identify feature types
    feature_eng = FeatureEngineer()
    df_engineered = feature_eng.transform(df)
    
    # Select features by removing non-feature columns
    feature_selector = FeatureSelector(drop_columns=non_features)
    X_selected = feature_selector.transform(df_engineered)
    
    # Identify feature types for preprocessing
    numeric_features = X_selected.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X_selected.select_dtypes(include=['object', 'category']).columns.tolist()
    
    logger.info(f"Numeric features: {len(numeric_features)}")
    logger.info(f"Categorical features: {len(categorical_features)}")
    
    # Create preprocessing pipeline with separate steps for numeric and categorical features
    preprocessor = ColumnTransformer(
        transformers=[
            # Numeric preprocessing: impute missing values with median, then scale
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), numeric_features),
            # Categorical preprocessing: impute missing values with most frequent, then one-hot encode
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ]), categorical_features)
        ]
    )
    
    # Build the final pipeline with all steps
    # This is the pipeline that will be saved and used for predictions
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
    
    return pipeline, df_engineered['log_price']

def train_and_save_model(pipeline, df, y, output_file='model.pkl'):
    """
    Train the model pipeline and save it to disk using dill serialization.
    """
    logger.info("Training model...")
    
    # Train the model on the input data and target
    pipeline.fit(df, y)
    
    logger.info("Saving model to disk...")
    
    # Save model to disk using dill for better serialization
    with open(output_file, 'wb') as f:
        dill.dump(pipeline, f)
    
    logger.info(f"Model saved to {output_file}")
    
    return output_file

def test_model(pipeline, df):
    """
    Test the model with a sample property to verify it works correctly.
    """
    # Create a sample property
    sample = df.iloc[[0]].copy()
    
    # Make prediction
    try:
        # Predict log price
        log_pred = pipeline.predict(sample)[0]
        # Convert back to original price scale
        pred_price = np.expm1(log_pred)
        
        # Log results
        logger.info(f"Sample property: {sample[['bedrooms', 'bathrooms', 'sqft_living']].iloc[0].to_dict()}")
        logger.info(f"Actual price: ${sample['price'].iloc[0]:.2f}")
        logger.info(f"Predicted price: ${pred_price:.2f}")
        
        return True
    except Exception as e:
        logger.error(f"Error testing model: {e}")
        return False

if __name__ == "__main__":
    # Load data from CSV
    df = load_csv_data()
    
    # Build model pipeline
    pipeline, y = build_model_pipeline(df)
    
    # Train and save model
    model_file = train_and_save_model(pipeline, df, y)
    
    # Test the model
    test_model(pipeline, df)
    
    logger.info(f"Model created from real data and saved to {model_file}")
    logger.info("You can now use this model with the prediction service by running:")
    logger.info("python prediction_service.py --model-file model.pkl") 