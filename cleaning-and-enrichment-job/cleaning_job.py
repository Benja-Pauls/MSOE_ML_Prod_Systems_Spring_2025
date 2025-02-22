import os
import psycopg
from marshmallow import Schema, fields, pre_load, validates, ValidationError

# Define the schema using marshmallow
class HomeSaleEventSchema(Schema):
    id = fields.Integer(required=True)
    # The cleaned_home_sale_events table has a "date" column, not "event_date"
    date = fields.Date(required=True)
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
        # Ensure we always have a 5-character zip, e.g. "00501"
        data['zipcode'] = data.get('zipcode', '').strip().zfill(5)
        # Convert school and population fields to int
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

    # Connect using psycopg (psycopg3)
    with psycopg.connect(conn_string) as conn:
        # Open a cursor
        with conn.cursor() as cur:
            # 1. Fetch unprocessed events
            cur.execute("""
                SELECT
                  id,
                  event_date,
                  data ->> 'zipcode' AS zipcode
                FROM raw_home_sale_events
                WHERE id NOT IN (SELECT id FROM processed_event_ids)
                ORDER BY event_date DESC
                LIMIT 100;
            """)
            events = cur.fetchall()  # list of tuples

            for event in events:
                # event is a tuple like (id, event_date, zipcode_string)
                event_id = event[0]
                raw_event_date = event[1]
                raw_zipcode = event[2]

                try:
                    # 2. Query population data
                    cur.execute("""
                        SELECT population
                        FROM cleaned_zipcode_populations
                        WHERE zipcode = %s
                    """, (raw_zipcode,))
                    population_data = cur.fetchone()

                    # 3. Query school data
                    cur.execute("""
                        SELECT high_schools,
                               middle_schools,
                               primary_schools,
                               other_schools,
                               unknown_schools,
                               total_schools
                        FROM cleaned_zipcode_public_schools
                        WHERE zipcode = %s
                    """, (raw_zipcode,))
                    school_data = cur.fetchone()

                    # If there's no matching row in population or school data, handle it
                    if population_data is None:
                        raise ValidationError(f"No population data for zipcode: {raw_zipcode}")
                    if school_data is None:
                        raise ValidationError(f"No school data for zipcode: {raw_zipcode}")

                    # 4. Build enriched dict
                    enriched_event = {
                        'id': event_id,
                        # The Marshmallow schema expects "date", not "event_date"
                        'date': raw_event_date,
                        'zipcode': raw_zipcode,
                        'population': population_data[0],
                        'high_schools': school_data[0],
                        'middle_schools': school_data[1],
                        'primary_schools': school_data[2],
                        'other_schools': school_data[3],
                        'unknown_schools': school_data[4],
                        'total_schools': school_data[5]
                    }

                    # 5. Validate with Marshmallow
                    schema = HomeSaleEventSchema()
                    validated_event = schema.load(enriched_event)

                    # 6. Insert into cleaned_home_sale_events
                    cur.execute("""
                        INSERT INTO cleaned_home_sale_events
                        (id, date, zipcode, population,
                         high_schools, middle_schools, primary_schools,
                         other_schools, unknown_schools, total_schools)
                        VALUES (
                          %(id)s,
                          %(date)s,
                          %(zipcode)s,
                          %(population)s,
                          %(high_schools)s,
                          %(middle_schools)s,
                          %(primary_schools)s,
                          %(other_schools)s,
                          %(unknown_schools)s,
                          %(total_schools)s
                        );
                    """, validated_event)

                    # 7. Mark event as processed
                    cur.execute("INSERT INTO processed_event_ids (id) VALUES (%s);", (event_id,))
                    conn.commit()

                except ValidationError as e:
                    print(f"Validation error for event {event_id}: {e}")
                    # Even if the row is invalid, mark it processed so we don’t retry next time
                    cur.execute("INSERT INTO processed_event_ids (id) VALUES (%s);", (event_id,))
                    conn.commit()

                except Exception as e:
                    print(f"Error processing event {event_id}: {e}")
                    # Roll back this record’s transaction so partial insert doesn't hang around
                    conn.rollback()

if __name__ == "__main__":
    check_environment()
    run_job()