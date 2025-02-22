import os
import psycopg
from marshmallow import Schema, fields, pre_load, validates, ValidationError

# Define the schema using marshmallow
class HomeSaleEventSchema(Schema):
    id = fields.Integer(required=True)
    event_date = fields.Date(required=True)
    zipcode = fields.String(required=True)
    population = fields.Integer(required=True)
    high_schools = fields.Integer(required=True)
    middle_schools = fields.Integer(required=True)
    primary_schools = fields.Integer(required=True)
    other_schools = fields.Integer(required=True)
    unknown_schools = fields.Integer(required=True)
    total_schools = fields.Integer(required=True)

    @pre_load
    def preprocess(self, data, **kwargs):
        data['zipcode'] = data.get('zipcode', '').strip().zfill(5)
        data['population'] = int(data.get('population', 0))
        data['high_schools'] = int(data.get('high_schools', 0))
        data['middle_schools'] = int(data.get('middle_schools', 0))
        data['primary_schools'] = int(data.get('primary_schools', 0))
        data['other_schools'] = int(data.get('other_schools', 0))
        data['unknown_schools'] = int(data.get('unknown_schools', 0))
        data['total_schools'] = int(data.get('total_schools', 0))
        return data

    @validates('population')
    def validate_population(self, value):
        if value < 0:
            raise ValidationError("Population cannot be negative")

def check_environment():
    required_vars = ['DB_USER', 'DB_PASSWORD', 'DB_HOST']
    for var in required_vars:
        if not os.getenv(var):
            raise EnvironmentError(f"Environment variable {var} is not set.")

def run_job():
    user = os.getenv('DB_USER')
    password = os.getenv('DB_PASSWORD')
    host = os.getenv('DB_HOST')
    conn_string = f"postgresql://{user}:{password}@{host}:5432/house_price_prediction_service"

    with psycopg.connect(conn_string) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, event_date, data->>'zipcode' AS zipcode FROM raw_home_sale_events
                WHERE id NOT IN (SELECT id FROM processed_event_ids)
                ORDER BY event_date DESC
                LIMIT 100;
            """)
            events = cur.fetchall()

            for event in events:
                try:
                    cur.execute("""
                        SELECT population FROM cleaned_zipcode_populations WHERE zipcode = %s
                    """, (event['zipcode'],))
                    population_data = cur.fetchone()

                    cur.execute("""
                        SELECT high_schools, middle_schools, primary_schools, other_schools, unknown_schools, total_schools
                        FROM cleaned_zipcode_public_schools WHERE zipcode = %s
                    """, (event['zipcode'],))
                    school_data = cur.fetchone()

                    enriched_event = {
                        'id': event['id'],
                        'event_date': event['event_date'],
                        'zipcode': event['zipcode'],
                        'population': population_data['population'],
                        'high_schools': school_data['high_schools'],
                        'middle_schools': school_data['middle_schools'],
                        'primary_schools': school_data['primary_schools'],
                        'other_schools': school_data['other_schools'],
                        'unknown_schools': school_data['unknown_schools'],
                        'total_schools': school_data['total_schools']
                    }

                    schema = HomeSaleEventSchema()
                    validated_event = schema.load(enriched_event)

                    cur.execute("""
                        INSERT INTO cleaned_home_sale_events (id, event_date, zipcode, population, high_schools, middle_schools, primary_schools, other_schools, unknown_schools, total_schools)
                        VALUES (%(id)s, %(event_date)s, %(zipcode)s, %(population)s, %(high_schools)s, %(middle_schools)s, %(primary_schools)s, %(other_schools)s, %(unknown_schools)s, %(total_schools)s);
                    """, validated_event)

                    cur.execute("INSERT INTO processed_event_ids (id) VALUES (%s);", (event['id'],))

                    conn.commit()

                except ValidationError as e:
                    print(f"Validation error for event {event['id']}: {e}")
                    cur.execute("INSERT INTO processed_event_ids (id) VALUES (%s);", (event['id'],))
                    conn.commit()

                except Exception as e:
                    print(f"Error processing event {event['id']}: {e}")
                    conn.rollback()

if __name__ == "__main__":
    check_environment()
    run_job()