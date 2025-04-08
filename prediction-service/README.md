# House Price Prediction Service

A RESTful service that predicts the sale price of homes based on their descriptions.

## Overview

This service:

1. Handles prediction requests for house prices
2. Queries the enrichment store for additional data (population and school counts)
3. Cleans and validates the input data using Marshmallow
4. Extracts features from the input data
5. Uses a trained machine learning model to make predictions
6. Returns a JSON response with the predicted house price

## Setup and Installation

### Prerequisites

- Python 3.12+
- PostgreSQL database with house price data and enrichment tables
- Required Python packages:
  - Flask
  - psycopg (with binary extras)
  - marshmallow
  - scikit-learn
  - dill
  - pandas
  - numpy

### Installation

1. Clone the repository
2. Install dependencies:
   ```
   pip install "psycopg[binary]" flask marshmallow scikit-learn dill pandas numpy
   ```

### Environment Variables

The service requires the following environment variables:

- `DB_USER`: PostgreSQL username
- `DB_PASSWORD`: PostgreSQL password
- `DB_HOST`: PostgreSQL host address

## Usage

### Running the Service

Run the service with:

```bash
python prediction_service.py --model-file model.pkl
```

Options:
- `--model-file`: Path to the serialized model file (required)
- `--port`: Port to run the service on (default: 8888)
- `--host`: Host to run the service on (default: 0.0.0.0)

### API Endpoints

#### `POST /predicted-home-value`

Predicts the value of a home based on provided attributes.

**Request Format:**

```json
{
  "property": {
    "sale_date": "2024-12-30",
    "sqft_living": 2000,
    "sqft_lot": 4000,
    "sqft_above": 500,
    "sqft_basement": 500,
    "sqft_living15": 2000,
    "sqft_lot15": 4000,
    "year_built": 1975,
    "year_renovated": null,
    "zipcode": "06106",
    "latitude": 35.1,
    "longitude": 37.1,
    "floors": 1.5,
    "waterfront": true,
    "bedrooms": 2,
    "bathrooms": 1.5
  }
}
```

**Response Format:**

```json
{
  "predicted_price": 250321.45
}
```

### Docker

#### Building the Docker Image

```bash
docker build --no-cache -t prediction-service .
```

#### Running in Docker

```bash
docker run \
  -e DB_USER=<username> \
  -e DB_PASSWORD=<password> \
  -e DB_HOST=<host> \
  -p 8888:8888 \
  prediction-service
```

## Model Creation and Testing

### Recreating the Model

We provide two scripts for model creation:

1. `recreate_model.py` - Recreates the model using real data from the PostgreSQL database
2. `create_test_model.py` - Creates a test model using synthetic data (useful for development)

To recreate the model using the actual database:

```bash
export DB_USER=<username>
export DB_PASSWORD=<password>
export DB_HOST=<host>
python recreate_model.py
```

To create a test model with synthetic data:

```bash
python create_test_model.py
```

### Testing the Model

Use the `test_model.py` script to test if the model works correctly:

```bash
python test_model.py
```

This script:
1. Loads the model
2. Creates a sample property input
3. Makes a prediction
4. Displays the result

## Testing the Service

Run unit tests with:

```bash
python test_prediction_service.py
```

For manual testing with curl:

```bash
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{
    "property": {
      "sale_date": "2024-12-30",
      "sqft_living": 2000,
      "sqft_lot": 4000,
      "sqft_above": 1500,
      "sqft_basement": 500,
      "zipcode": "98001",
      "latitude": 47.3,
      "longitude": -122.2,
      "floors": 2,
      "waterfront": false,
      "bedrooms": 3,
      "bathrooms": 2.5
    }
  }' \
  http://localhost:8888/predicted-home-value
```

## Troubleshooting

If you encounter issues:

1. Check that all required environment variables are set
2. Verify the model file exists and is accessible
3. Check database connectivity
4. Review logs for detailed error information
5. Ensure input data is properly formatted

If the model is corrupt or encountering errors, try recreating it using one of the provided scripts.