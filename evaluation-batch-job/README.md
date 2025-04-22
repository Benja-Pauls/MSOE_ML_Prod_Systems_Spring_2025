# Evaluation Batch Job

This directory contains the scripts and Docker setup for the evaluation batch job. The job processes cleaned home sale events, gets predictions from the prediction service, and stores the results in a PostgreSQL database for monitoring model performance.

## Database Schema

### `evaluation_results`

- **id**: `SERIAL` - Primary key, auto-incremented.
- **event_id**: `BIGINT` - ID of the home sale event.
- **sale_date**: `DATE` - Date of the sale.
- **true_price**: `FLOAT` - Actual sale price.
- **predicted_price**: `FLOAT` - Predicted sale price from the model.
- **created_at**: `TIMESTAMP` - When the evaluation was performed.

### `evaluated_event_ids`

- **id**: `BIGINT` - Primary key, ID of processed events.

## Instructions

### Running the Script Standalone

1. **Set Environment Variables**: Ensure the following environment variables are set:
   - `DB_USER`: PostgreSQL username
   - `DB_PASSWORD`: PostgreSQL password
   - `DB_HOST`: PostgreSQL host address
   - `PREDICTION_SERVICE_URL`: URL of the prediction service (e.g., http://prediction-service:8888)

2. **Run the Script**: Execute the `evaluation_job.py` script directly:
   ```bash
   python3 evaluation_job.py
   ```

### Building and Running the Docker Image

1. **Build the Docker Image**: From the `evaluation-batch-job` directory, build the Docker image:
   ```bash
   docker build --no-cache -t evaluation-batch-job .
   ```

2. **Run the Docker Container**: Use the following command to run the container in detached mode:
   ```bash
   docker run -d \
     --name evaluation-batch-job-container \
     --network home-sale-event-system \
     -e DB_USER=postgres \
     -e DB_PASSWORD=psql-password \
     -e DB_HOST=postgres \
     -e PREDICTION_SERVICE_URL=http://prediction-service:8888 \
     evaluation-batch-job
   ```

3. **Check Logs**: To view the output logs of the running container:
   ```bash
   docker logs evaluation-batch-job-container
   ```

## Notes

- The job runs every 15 minutes
- It processes up to 100 unprocessed events per run
- Results are stored in the evaluation_results table
- Processed event IDs are tracked in the evaluated_event_ids table
