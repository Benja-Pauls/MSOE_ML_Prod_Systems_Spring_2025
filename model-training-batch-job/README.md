# Model Training Batch Job

This component implements a batch job that periodically trains and evaluates a machine learning model for house price prediction, then stores the trained model in the database.

## Overview

The training batch job performs the following tasks:

1. Retrieves cleaned and joined data from the database
2. Performs a train-test split
3. Trains the feature extraction and model pipeline on the training data
4. Evaluates the model using the testing data
5. Re-trains the pipeline using all of the data
6. Deposits the trained pipeline into a PostgreSQL table along with evaluation metrics

## Database Schema

The job creates and uses the following database table:

```sql
CREATE TABLE IF NOT EXISTS trained_models (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(100),
    model_data BYTEA,             -- Serialized model pipeline
    rmse FLOAT,                   -- Root Mean Squared Error
    r2_score FLOAT,               -- R-squared score
    mape FLOAT,                   -- Mean Absolute Percentage Error
    training_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    model_version VARCHAR(50),    -- Version identifier
    features_used TEXT[]          -- Array of feature names used in the model
)
```

## Components

- **database.py**: Handles database connections and operations
- **train.py**: Contains model training and evaluation logic
- **training_job.py**: Main script for the training job
- **job_runner.py**: Runs the training job at specified intervals

## Model Pipeline

The model pipeline consists of:

1. **Feature Engineering**: Creates derived features from raw data
2. **Feature Selection**: Removes unnecessary columns
3. **Preprocessing**: Handles missing values, scales numeric features, and one-hot encodes categorical features
4. **Regression Model**: Uses RandomForest with optimized hyperparameters

## Running the Job

### Environment Variables

- `DB_USER`: Database username
- `DB_PASSWORD`: Database password
- `DB_HOST`: Database host
- `TRAINING_INTERVAL`: Training interval in hours (default: 24)

### Using Docker

```bash
# Set environment variables
export DB_USER=your_db_user
export DB_PASSWORD=your_db_password
export DB_HOST=your_db_host
export TRAINING_INTERVAL=24  # Optional, defaults to 24 hours

# Build and run the container
./build_and_run.sh
```

### Using Docker Compose

```bash
# Set environment variables
export DB_USER=your_db_user
export DB_PASSWORD=your_db_password
export DB_HOST=your_db_host
export TRAINING_INTERVAL=24  # Optional, defaults to 24 hours

# Start the container
docker-compose up -d
```

### Manual Execution

```bash
# Install dependencies
pip install -r requirements.txt

# Run the job
python -m src.training_job  # Run a single training job
python -m src.job_runner    # Run the job scheduler
```

## Viewing Logs

```bash
# View container logs
docker logs -f model-training-job
``` 