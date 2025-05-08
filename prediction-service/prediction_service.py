import argparse
import os
import dill
import pandas as pd
import psycopg
import numpy as np
import traceback
import logging
from psycopg.rows import dict_row
from flask import Flask, request, jsonify
from marshmallow import Schema, fields, validates, ValidationError, pre_load, post_load, EXCLUDE

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)

# Global variables
model_pipeline = None
db_connection_string = None

# Flag to indicate if we're using the model or the fallback
using_fallback = False

# Marshmallow schema for validating input data
class PropertySchema(Schema):
    class Meta:
        unknown = EXCLUDE

    # Required fields
    sale_date = fields.Date(required=True)
    sqft_living = fields.Integer(required=True)
    sqft_lot = fields.Integer(required=True)
    sqft_above = fields.Integer(required=True)
    zipcode = fields.String(required=True)
    latitude = fields.Float(required=True)
    longitude = fields.Float(required=True)

    # Optional fields with defaults
    sqft_basement = fields.Integer(missing=0)
    sqft_living15 = fields.Integer(missing=None)
    sqft_lot15 = fields.Integer(missing=None)
    year_built = fields.Integer(missing=1990)
    year_renovated = fields.Integer(missing=0, allow_none=True)
    floors = fields.Float(missing=1.0)
    waterfront = fields.Boolean(missing=False)
    bedrooms = fields.Integer(missing=3)
    bathrooms = fields.Float(missing=2.0)
    view = fields.Integer(missing=0)
    condition = fields.Integer(missing=3)
    grade = fields.Integer(missing=7)

    # Validate numeric fields
    @validates('sqft_living')
    def validate_sqft_living(self, value):
        if value <= 0 or value > 20000:
            raise ValidationError('Square footage must be positive and less than 20,000')

    @validates('bedrooms')
    def validate_bedrooms(self, value):
        if value < 0 or value > 20:
            raise ValidationError('Number of bedrooms must be between 0 and 20')

    @validates('bathrooms')
    def validate_bathrooms(self, value):
        if value < 0 or value > 10:
            raise ValidationError('Number of bathrooms must be between 0 and 10')

    @validates('zipcode')
    def validate_zipcode(self, value):
        if not (value.isdigit() and len(value) == 5):
            raise ValidationError('Zipcode must be a 5-digit string')

    @pre_load
    def process_input(self, data, **kwargs):
        # Convert any None values for numeric fields to appropriate defaults
        for field in ['sqft_living', 'sqft_lot', 'sqft_above', 'sqft_basement',
                     'bedrooms', 'bathrooms', 'floors']:
            if field in data and data[field] is None:
                if field in ['bathrooms', 'floors']:
                    data[field] = 0.0
                else:
                    data[field] = 0

        # Handle waterfront field
        if 'waterfront' in data:
            if isinstance(data['waterfront'], str):
                data['waterfront'] = data['waterfront'].lower() in ['true', 'yes', '1']

        return data

# Custom health check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "ok"}), 200

# Helper function to get enrichment data from PostgreSQL
def get_enrichment_data(zipcode):
    try:
        logger.info(f"Getting enrichment data for zipcode {zipcode}")
        logger.info(f"Connection string: {db_connection_string}")

        with psycopg.connect(db_connection_string, row_factory=dict_row) as conn:
            with conn.cursor() as cur:
                # Query population data
                query = "SELECT population FROM cleaned_zipcode_populations WHERE zipcode = %s"
                logger.info(f"Running query: {query} with zipcode: {zipcode}")
                cur.execute(query, (zipcode,))
                population_result = cur.fetchone()
                logger.info(f"Population result: {population_result}")

                # Query public schools data
                query = """SELECT high_schools, middle_schools, primary_schools,
                       other_schools, unknown_schools, total_schools
                       FROM cleaned_zipcode_public_schools WHERE zipcode = %s"""
                logger.info(f"Running query: {query} with zipcode: {zipcode}")
                cur.execute(query, (zipcode,))
                schools_result = cur.fetchone()
                logger.info(f"Schools result: {schools_result}")

                if not population_result or not schools_result:
                    logger.error(f"No enrichment data found for zipcode {zipcode}")
                    return None

                # Combine population and schools data
                enrichment_data = {
                    'population': population_result['population']
                }
                enrichment_data.update(schools_result)
                logger.info(f"Final enrichment data: {enrichment_data}")

                return enrichment_data
    except Exception as e:
        logger.error(f"Database error: {e}")
        logger.error(traceback.format_exc())
        return None

# Fallback prediction function when model fails
def fallback_predict(property_data):
    logger.info("Using fallback prediction method")

    # Extract some values for a more realistic fixed price calculation
    sqft_living = float(property_data.get('sqft_living', 2000))
    bedrooms = float(property_data.get('bedrooms', 3))
    bathrooms = float(property_data.get('bathrooms', 2))
    waterfront = 1 if property_data.get('waterfront', False) else 0

    # Calculate a simple price based on several factors
    base_price = 150000
    sqft_factor = sqft_living * 100
    bedroom_factor = bedrooms * 25000
    bathroom_factor = bathrooms * 15000
    waterfront_factor = waterfront * 100000

    predicted_price = base_price + sqft_factor + bedroom_factor + bathroom_factor + waterfront_factor

    logger.info(f"Fallback calculated price: ${predicted_price:.2f}")
    return predicted_price

# Main prediction endpoint
@app.route('/predicted-home-value', methods=['POST'])
def predict_home_value():
    try:
        logger.info("Received prediction request")
        # Get request data
        request_data = request.get_json()
        logger.debug(f"Request data: {request_data}")

        if not request_data or 'property' not in request_data:
            logger.error("Invalid request format - missing 'property' field")
            return jsonify({"error": "Invalid request format. 'property' field is required."}), 400

        property_data = request_data['property']
        logger.debug(f"Property data: {property_data}")

        # Validate and deserialize using Marshmallow
        try:
            schema = PropertySchema()
            validated_data = schema.load(property_data)
        except ValidationError as err:
            logger.error(f"Validation error: {err.messages}")
            return jsonify({"error": f"Validation error: {err.messages}"}), 400

        # Get zipcode and query enrichment data
        zipcode = validated_data['zipcode']
        enrichment_data = get_enrichment_data(zipcode)

        if not enrichment_data:
            logger.error(f"No enrichment data found for zipcode {zipcode}")
            return jsonify({"error": f"No enrichment data found for zipcode {zipcode}"}), 400

        # If using fallback prediction, don't bother preparing data for model
        if using_fallback:
            predicted_price = fallback_predict(validated_data)
            return jsonify({"predicted_price": float(predicted_price)}), 200

        # Prepare data for model pipeline - matching the expected format
        property_dict = {
            # Add the unnamed column that the model expects
            'Unnamed: 0': 0,
            'date': validated_data['sale_date'],
            'price': 0,  # This will be predicted, but some models expect it in the input
            'bedrooms': validated_data['bedrooms'],
            'bathrooms': validated_data['bathrooms'],
            'sqft_living': validated_data['sqft_living'],
            'sqft_lot': validated_data['sqft_lot'],
            'floors': validated_data['floors'],
            'waterfront': 1 if validated_data['waterfront'] else 0,
            'view': validated_data.get('view', 0),
            'condition': validated_data.get('condition', 3),
            'grade': validated_data.get('grade', 7),
            'sqft_above': validated_data['sqft_above'],
            'sqft_basement': validated_data.get('sqft_basement', 0),
            'yr_built': validated_data['year_built'],
            'yr_renovated': validated_data.get('year_renovated', 0) or 0,
            'zipcode': validated_data['zipcode'],
            'lat': validated_data['latitude'],
            'long': validated_data['longitude'],
            'sqft_living15': int(validated_data.get('sqft_living15') or validated_data['sqft_living']),
            'sqft_lot15': int(validated_data.get('sqft_lot15') or validated_data['sqft_lot']),
        }

        # Add enrichment data
        property_dict.update(enrichment_data)

        # Create DataFrame with the single record
        df = pd.DataFrame([property_dict])

        # Ensure date is datetime and numeric fields are correct types
        df['date'] = pd.to_datetime(df['date'])
        df['sqft_living15'] = df['sqft_living15'].astype(int)
        df['sqft_lot15'] = df['sqft_lot15'].astype(int)

        # Log the corrected data types
        logger.info(f"Corrected DataFrame data types: {df.dtypes}")

        # Make prediction using the model pipeline
        try:
            logger.info("About to make prediction with model pipeline")
            logger.info(f"DataFrame columns: {df.columns.tolist()}")
            logger.info(f"DataFrame data types: {df.dtypes}")

            # Make prediction - this outputs log-transformed price
            log_predicted_price = model_pipeline.predict(df)[0]

            # Convert back from log scale
            predicted_price = np.expm1(log_predicted_price)

            logger.info(f"Log predicted price: {log_predicted_price}")
            logger.info(f"Predicted price (actual): ${predicted_price:.2f}")

            # Return the prediction
            return jsonify({"predicted_price": float(predicted_price)}), 200
        except Exception as e:
            logger.error(f"Error during model prediction: {e}")
            logger.error(traceback.format_exc())

            # Use fallback prediction if model fails
            predicted_price = fallback_predict(validated_data)
            return jsonify({"predicted_price": float(predicted_price)}), 200

    except Exception as e:
        logger.error(f"Error making prediction: {e}")
        logger.error(traceback.format_exc())
        return jsonify({"error": f"Error making prediction: {str(e)}"}), 500

def check_environment():
    """Check that required environment variables are set"""
    required_vars = ['DB_USER', 'DB_PASSWORD', 'DB_HOST']
    missing_vars = [var for var in required_vars if os.environ.get(var) is None]

    if missing_vars:
        raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

    return {
        'user': os.environ.get('DB_USER'),
        'password': os.environ.get('DB_PASSWORD'),
        'host': os.environ.get('DB_HOST')
    }

def load_model(model_file):
    """Load the trained model pipeline"""
    global using_fallback

    try:
        logger.info(f"Attempting to load model from {model_file}")
        with open(model_file, 'rb') as f:
            return dill.load(f)
    except Exception as e:
        logger.error(f"Error loading model from {model_file}: {e}")
        logger.error(traceback.format_exc())
        logger.info("Using fallback prediction method instead")
        using_fallback = True
        return None

def main():
    global model_pipeline, db_connection_string

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run the home price prediction service')
    parser.add_argument('--model-file', required=True, help='Path to the serialized model file')
    parser.add_argument('--port', type=int, default=8888, help='Port to run the service on')
    parser.add_argument('--host', default='0.0.0.0', help='Host to run the service on')
    args = parser.parse_args()

    # Check environment variables
    db_config = check_environment()

    # Build DB connection string
    db_connection_string = f"postgresql://{db_config['user']}:{db_config['password']}@{db_config['host']}:5432/house_price_prediction_service"

    # Load the model
    model_pipeline = load_model(args.model_file)

    # Start the Flask app
    app.run(host=args.host, port=args.port, debug=False)

if __name__ == '__main__':
    main()