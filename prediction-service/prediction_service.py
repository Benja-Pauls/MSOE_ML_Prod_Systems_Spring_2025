import argparse
import os
import dill
import pandas as pd
import psycopg
from psycopg.rows import dict_row
from flask import Flask, request, jsonify
from marshmallow import Schema, fields, validates, ValidationError, pre_load, post_load, EXCLUDE
import logging
import traceback

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)

# Global variables
model_pipeline = None
db_connection_string = None

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
    year_built = fields.Integer(missing=None)
    year_renovated = fields.Integer(missing=None, allow_none=True)
    floors = fields.Float(missing=1.0)
    waterfront = fields.Boolean(missing=False)
    bedrooms = fields.Integer(missing=3)
    bathrooms = fields.Float(missing=2.0)

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
            return jsonify({"error": f"Validation error: {err.messages}"}), 400
        
        # Get zipcode and query enrichment data
        zipcode = validated_data['zipcode']
        enrichment_data = get_enrichment_data(zipcode)
        
        if not enrichment_data:
            return jsonify({"error": f"No enrichment data found for zipcode {zipcode}"}), 400
        
        # Combine property and enrichment data
        full_data = {**validated_data, **enrichment_data}
        
        # Create DataFrame with the single record
        df = pd.DataFrame([full_data])
        
        # Make prediction using the model pipeline
        predicted_price = model_pipeline.predict(df)[0]
        
        # Return the prediction
        return jsonify({"predicted_price": float(predicted_price)}), 200
        
    except Exception as e:
        logger.error(f"Error making prediction: {e}")
        logger.error(traceback.format_exc())
        return jsonify({"error": f"Error making prediction: {str(e)}"}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "ok"}), 200

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
    try:
        with open(model_file, 'rb') as f:
            return dill.load(f)
    except Exception as e:
        raise ValueError(f"Error loading model from {model_file}: {e}")

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

# Set up error handling
@app.errorhandler(Exception)
def handle_exception(e):
    logger.error(f"Unhandled exception: {str(e)}")
    logger.error(traceback.format_exc())
    return jsonify({"error": f"Server error: {str(e)}"}), 500

if __name__ == '__main__':
    main()