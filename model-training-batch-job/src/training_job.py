#!/usr/bin/env python3
"""
Model Training Batch Job

This script periodically:
1. Grabs cleaned and joined data from the database
2. Performs a train-test split
3. Trains the feature extraction and model pipeline on the training data
4. Evaluates the model using the testing data
5. Re-trains the pipeline using all of the data
6. Deposits the trained pipeline into a PostgreSQL table along with evaluation metrics

Usage:
    python training_job.py [--test]
    
    --test    Run in test mode with limited data sample for quick testing
"""

import os
import sys
import logging
import psycopg
import argparse

# Add the current directory to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from src.database import check_environment, get_connection_string, create_tables, get_training_data, store_model
from src.train import train_test_model, train_final_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_training_job(test_mode=False):
    """
    Execute the model training batch job.
    
    Args:
        test_mode (bool): If True, run in test mode with limited data
    """
    logger.info(f"Starting model training batch job (test mode: {test_mode})")
    
    try:
        # Check environment variables
        check_environment()
        
        # Get database connection string
        conn_string = get_connection_string()
        
        # Connect to database
        with psycopg.connect(conn_string) as conn:
            logger.info("Connected to database")
            
            # Create necessary tables if they don't exist
            create_tables(conn)
            
            # Get training data
            df = get_training_data(conn, limit=1000 if test_mode else None)
            
            if len(df) == 0:
                logger.error("No training data retrieved from database")
                return
                
            logger.info(f"Retrieved {len(df)} records for training")
            
            # Train and evaluate model
            pipeline, metrics, features = train_test_model(df)
            
            # Train final model on all data
            final_pipeline, features = train_final_model(df)
            
            # Store model in database
            model_id = store_model(
                conn=conn,
                model=final_pipeline,
                model_name="house_price_prediction_model",
                metrics=metrics,
                features=features
            )
            
            logger.info(f"Model training job completed successfully. Model ID: {model_id}")
            
    except Exception as e:
        logger.error(f"Error in training job: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run model training batch job')
    parser.add_argument('--test', action='store_true', help='Run in test mode with limited data')
    args = parser.parse_args()
    
    run_training_job(test_mode=args.test) 