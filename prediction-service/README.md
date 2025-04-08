# Home Price Prediction Service

This service provides a RESTful API for predicting home prices based on property attributes.

## API Endpoints

### POST /predicted-home-value

Predicts the price of a home based on its attributes.

#### Request Format

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
    "zipcode": "98001",
    "latitude": 35.1,
    "longitude": 37.1,
    "floors": 1.5,
    "waterfront": true,
    "bedrooms": 2,
    "bathrooms": 1.5
  }
}
```

#### Response Format

```json
{
  "predicted_price": 250321.45
}
```

## Running the Service

### Running Directly

1. Set environment variables:
```bash
export DB_USER=postgres
export DB_PASSWORD=psql-password
export DB_HOST=localhost
```

2. Run the service:
```bash
python prediction_service.py --model-file /path/to/model.pkl
```

### Running in Docker

1. Build the Docker image:
```bash
docker build --no-cache -t prediction-service .
```

2. Run the Docker container:
```bash
docker run -d \
  --name prediction-service-container \
  --network home-sale-event-system \
  -e DB_USER=postgres \
  -e DB_PASSWORD=psql-password \
  -e DB_HOST=postgres \
  -p 8888:8888 \
  prediction-service
```

3. Check logs:
```bash
docker logs prediction-service-container
```

## Running Tests

### Running Tests Directly

1. Set environment variables:
```bash
export TEST_HOST=localhost
export TEST_PORT=8888
```

2. Run the tests:
```bash
python test_prediction_service.py
```

### Running Tests Against Dockerized Service

1. Make sure the service is running in Docker
2. Run the tests:
```bash
export TEST_HOST=localhost
export TEST_PORT=8888
python test_prediction_service.py
```
```

### Deployment Steps

Now let's put everything together:

1. Create the prediction-service directory and place all the files there:
```bash
mkdir -p prediction-service
cd prediction-service
# Create prediction_service.py, test_prediction_service.py, Dockerfile, and README.md with the contents provided
```

2. Make sure you have the trained model file (model.pkl) in the directory. This should be the serialized model pipeline from Project 4.

3. Build the Docker image:
```bash
docker build --no-cache -t prediction-service .
```

4. Run the Docker container:
```bash
docker run -d \
  --name prediction-service-container \
  --network home-sale-event-system \
  -e DB_USER=postgres \
  -e DB_PASSWORD=psql-password \
  -e DB_HOST=postgres \
  -p 8888:8888 \
  prediction-service
```

5. Test the service:
```bash
export TEST_HOST=localhost
export TEST_PORT=8888
python test_prediction_service.py
```

6. Test with curl:
```bash
curl -H "Content-Type: application/json" -d '{"property": {"sale_date": "2024-12-30","sqft_living": 2000,"sqft_lot": 4000,"sqft_above": 500,"zipcode": "98001","latitude": 35.1,"longitude": 37.1}}' http://localhost:8888/predicted-home-value
```

You've now completed all parts of Project 5!