import os
import psycopg
import pandas as pd
import pickle
import logging
from psycopg.rows import dict_row
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

def get_connection_string():
    """Build and return the PostgreSQL connection string"""
    env = check_environment()
    return f"postgresql://{env['user']}:{env['password']}@{env['host']}:5432/house_price_prediction_service"

def create_tables(conn):
    """Create the trained_models table if it doesn't exist"""
    with conn.cursor() as cur:
        # Create trained models table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS trained_models (
                id SERIAL PRIMARY KEY,
                model_name VARCHAR(100),
                model_data BYTEA,
                rmse FLOAT,
                r2_score FLOAT,
                mape FLOAT,
                training_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                model_version VARCHAR(50),
                features_used TEXT[]
            )
        """)
        
        # Create indices
        cur.execute("CREATE INDEX IF NOT EXISTS idx_model_training_date ON trained_models(training_date)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_model_name ON trained_models(model_name)")
        
        conn.commit()
        logger.info("Database tables created or verified")

def get_training_data(conn, limit=None):
    """
    Retrieve cleaned and joined data from the database
    
    Args:
        conn: Database connection
        limit (int, optional): Limit the number of rows returned (for testing)
    
    Returns:
        pandas.DataFrame: Training data
    """
    logger.info(f"Retrieving training data from database{' (limited sample)' if limit else ''}...")
    
    query = """
        SELECT 
            date, price, bedrooms, bathrooms, sqft_living, sqft_lot, 
            floors, waterfront, view, condition, grade, sqft_above, 
            sqft_basement, yr_built, yr_renovated, zipcode, lat, long, 
            sqft_living15, population, high_schools, middle_schools, 
            primary_schools, other_schools, unknown_schools, total_schools
        FROM 
            cleaned_home_sale_events
        ORDER BY 
            date DESC
    """
    
    # Add limit if specified
    if limit:
        query += f" LIMIT {limit}"
    
    try:
        df = pd.read_sql_query(query, conn)
        logger.info(f"Retrieved {len(df)} records for training")
        return df
    except Exception as e:
        logger.error(f"Error retrieving training data: {e}")
        raise

def store_model(conn, model, model_name, metrics, features):
    """Store the trained model and metrics in the database"""
    logger.info("Storing trained model in database...")
    
    try:
        # Serialize the model
        model_bytes = pickle.dumps(model)
        
        # Insert model into database
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO trained_models 
                (model_name, model_data, rmse, r2_score, mape, model_version, features_used)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                RETURNING id
            """, (
                model_name,
                model_bytes,
                metrics['rmse'],
                metrics['r2'],
                metrics['mape'],
                datetime.now().strftime("%Y%m%d_%H%M"),
                features
            ))
            
            model_id = cur.fetchone()[0]
            conn.commit()
            
            logger.info(f"Model stored successfully with ID: {model_id}")
            return model_id
    except Exception as e:
        logger.error(f"Error storing model: {e}")
        conn.rollback()
        raise 