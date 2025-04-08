import dill
import pandas as pd
import numpy as np
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_model(model_file='model.pkl'):
    """Load the serialized model"""
    try:
        with open(model_file, 'rb') as f:
            model = dill.load(f)
        logger.info(f"Model loaded successfully from {model_file}")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None

def create_sample_input():
    """Create a sample property input like what the API would receive"""
    # Create a sample input similar to what would come through the API
    property_dict = {
        # Include the unnamed column that the model expects
        'Unnamed: 0': 0,
        'date': datetime.now().strftime('%Y-%m-%d'),
        'price': 0,  # This will be predicted, but model expects it
        'bedrooms': 3,
        'bathrooms': 2.5,
        'sqft_living': 2200,
        'sqft_lot': 5000,
        'floors': 2.0,
        'waterfront': 0,
        'view': 0,
        'condition': 3,
        'grade': 7,
        'sqft_above': 1700,
        'sqft_basement': 500,
        'yr_built': 1985,
        'yr_renovated': 0,
        'zipcode': '98001',  # This would usually come from user input
        'lat': 47.3,
        'long': -122.2,
        'sqft_living15': 2100,
        'sqft_lot15': 4800,
        # Enrichment data that would come from database
        'population': 20000,
        'high_schools': 2,
        'middle_schools': 3,
        'primary_schools': 5,
        'other_schools': 1,
        'unknown_schools': 0,
        'total_schools': 11
    }
    
    # Convert to DataFrame (model expects DataFrame input)
    df = pd.DataFrame([property_dict])
    
    # Ensure date is datetime
    df['date'] = pd.to_datetime(df['date'])
    
    return df

def test_prediction(model, df):
    """Test making a prediction with the model"""
    try:
        # Display the input
        logger.info(f"Input property: {df[['bedrooms', 'bathrooms', 'sqft_living', 'zipcode']].iloc[0].to_dict()}")
        
        # Make prediction
        log_price = model.predict(df)[0]
        price = np.expm1(log_price)
        
        logger.info(f"Log predicted price: {log_price:.4f}")
        logger.info(f"Predicted price: ${price:.2f}")
        
        return price
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

if __name__ == "__main__":
    # Load the model
    model = load_model()
    
    if model:
        # Create sample input
        sample_df = create_sample_input()
        
        # Test prediction
        predicted_price = test_prediction(model, sample_df)
        
        if predicted_price:
            logger.info("Test successful!")
        else:
            logger.error("Test failed!") 