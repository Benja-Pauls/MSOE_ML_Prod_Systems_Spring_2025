import os
import psycopg
import requests
import logging
from datetime import datetime
from psycopg.rows import dict_row

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_environment():
    """Check that required environment variables are set"""
    required_vars = ['DB_USER', 'DB_PASSWORD', 'DB_HOST', 'PREDICTION_SERVICE_URL']
    missing_vars = [var for var in required_vars if os.environ.get(var) is None]
    
    if missing_vars:
        raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
    
    return {
        'user': os.environ.get('DB_USER'),
        'password': os.environ.get('DB_PASSWORD'),
        'host': os.environ.get('DB_HOST'),
        'prediction_url': os.environ.get('PREDICTION_SERVICE_URL')
    }

def create_tables(conn):
    """Create the evaluation tables if they don't exist"""
    with conn.cursor() as cur:
        # Create evaluation results table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS evaluation_results (
                id SERIAL PRIMARY KEY,
                event_id BIGINT NOT NULL,
                sale_date DATE NOT NULL,
                true_price FLOAT NOT NULL,
                predicted_price FLOAT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create processed events table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS evaluated_event_ids (
                id BIGINT PRIMARY KEY
            )
        """)
        
        # Create indices
        cur.execute("CREATE INDEX IF NOT EXISTS idx_evaluation_event_id ON evaluation_results(event_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_evaluation_sale_date ON evaluation_results(sale_date)")
        
        conn.commit()

def get_unprocessed_events(conn):
    """Get events that haven't been evaluated yet"""
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute("""
            SELECT id, event_date, data->>'price' as price,
                   data->>'sqft_living' as sqft_living,
                   data->>'sqft_lot' as sqft_lot,
                   data->>'sqft_above' as sqft_above,
                   data->>'sqft_basement' as sqft_basement,
                   data->>'zipcode' as zipcode,
                   data->>'lat' as lat,
                   data->>'long' as long,
                   data->>'floors' as floors,
                   data->>'waterfront' as waterfront,
                   data->>'bedrooms' as bedrooms,
                   data->>'bathrooms' as bathrooms
            FROM raw_home_sale_events
            WHERE id NOT IN (SELECT id FROM evaluated_event_ids)
            ORDER BY event_date DESC
            LIMIT 100
        """)
        return cur.fetchall()

def get_prediction(property_data, prediction_url):
    """Get prediction from the prediction service"""
    try:
        response = requests.post(
            f"{prediction_url}/predicted-home-value",
            json={"property": property_data},
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        return response.json()["predicted_price"]
    except Exception as e:
        logger.error(f"Error getting prediction: {e}")
        return None

def store_evaluation_result(conn, event_id, sale_date, true_price, predicted_price):
    """Store evaluation result in the database"""
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO evaluation_results (event_id, sale_date, true_price, predicted_price)
            VALUES (%s, %s, %s, %s)
        """, (event_id, sale_date, true_price, predicted_price))
        
        cur.execute("""
            INSERT INTO evaluated_event_ids (id)
            VALUES (%s)
        """, (event_id,))
        
        conn.commit()

def run_job():
    """Run the evaluation job"""
    logger.info("Starting evaluation job")
    
    # Get environment variables
    env = check_environment()
    
    # Build connection string
    conn_string = f"postgresql://{env['user']}:{env['password']}@{env['host']}:5432/house_price_prediction_service"
    
    try:
        # Connect to database
        with psycopg.connect(conn_string) as conn:
            # Create tables if they don't exist
            create_tables(conn)
            
            # Get unprocessed events
            events = get_unprocessed_events(conn)
            logger.info(f"Found {len(events)} unprocessed events")
            
            # Process each event
            for event in events:
                try:
                    # Prepare property data for prediction
                    property_data = {
                        "sale_date": event['event_date'].strftime('%Y-%m-%d'),
                        "sqft_living": int(event['sqft_living']),
                        "sqft_lot": int(event['sqft_lot']),
                        "sqft_above": int(event['sqft_above']),
                        "sqft_basement": int(event['sqft_basement']),
                        "zipcode": event['zipcode'],
                        "latitude": float(event['lat']),
                        "longitude": float(event['long']),
                        "floors": float(event['floors']),
                        "waterfront": event['waterfront'] == '1',
                        "bedrooms": int(event['bedrooms']),
                        "bathrooms": float(event['bathrooms'])
                    }
                    
                    # Get prediction
                    predicted_price = get_prediction(property_data, env['prediction_url'])
                    
                    if predicted_price is not None:
                        # Store result
                        store_evaluation_result(
                            conn,
                            event['id'],
                            event['event_date'],
                            float(event['price']),
                            predicted_price
                        )
                        logger.info(f"Processed event {event['id']}")
                    else:
                        logger.error(f"Failed to get prediction for event {event['id']}")
                
                except Exception as e:
                    logger.error(f"Error processing event {event['id']}: {e}")
                    continue
            
            logger.info("Evaluation job completed")
    
    except Exception as e:
        logger.error(f"Error in evaluation job: {e}")
        raise

if __name__ == "__main__":
    run_job()
