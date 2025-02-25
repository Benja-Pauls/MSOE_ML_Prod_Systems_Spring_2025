# Cleaning and Enrichment Job

This directory contains the scripts and Docker setup for the cleaning and enrichment batch job. The job processes raw home sale events, enriches them with population and public school data, and stores the cleaned data in a PostgreSQL database.

## Instructions

### Running the Script Standalone

1. **Set Environment Variables**: Ensure the following environment variables are set for database access:
   - `DB_USER`: Your PostgreSQL username
   - `DB_PASSWORD`: Your PostgreSQL password
   - `DB_HOST`: The host address of your PostgreSQL server

2. **Run the Script**: Execute the `cleaning_job.py` script directly:
   ```bash
   python3 cleaning_job.py
   ```

### Building and Running the Docker Image

1. **Build the Docker Image**: From the `cleaning-and-enrichment-job` directory, build the Docker image:
   ```bash
   docker build --no-cache -t cleaning-and-enrichment-job .
   ```

2. **Run the Docker Container**: Use the following command to run the container in detached mode:
   ```bash
   docker run -d \
     --name cleaning-and-enrichment-job-container \
     --network="host" \
     -e DB_USER=postgres \
     -e DB_PASSWORD=psql-password \
     -e DB_HOST=localhost \
     cleaning-and-enrichment-job
   ```

3. **Check Logs**: To view the output logs of the running container:
   ```bash
   docker logs cleaning-and-enrichment-job-container
   ```
## Database Schemas

### `raw_home_sale_events`

- **id**: `integer` - Primary key, auto-incremented.
- **data**: `jsonb` - JSON data containing raw event details.
- **event_date**: `date` - Date of the event.
- **lat**: `float` - Latitude coordinate.
- **long**: `float` - Longitude coordinate.
- **price**: `float` - Sale price of the house.
- **bedrooms**: `integer` - Number of bedrooms.
- **bathrooms**: `float` - Number of bathrooms.
- **sqft_living**: `integer` - Square footage of living space.
- **sqft_lot**: `integer` - Square footage of the lot.
- **floors**: `float` - Number of floors.
- **waterfront**: `integer` - Waterfront view indicator.
- **view**: `integer` - Quality of the view.
- **condition**: `integer` - Condition of the house.
- **grade**: `integer` - Construction grade.
- **sqft_above**: `integer` - Square footage above ground.
- **yr_built**: `integer` - Year built.
- **yr_renovated**: `integer` - Year renovated.
- **sqft_living15**: `integer` - Living space of 15 nearest neighbors.

### `cleaned_home_sale_events`

- **id**: `bigint` - Primary key, corresponds to the raw event ID.
- **event_date**: `date` - Date of the event.
- **zipcode**: `character varying(10)` - Zip code of the event location.
- **lat**: `float` - Latitude coordinate.
- **long**: `float` - Longitude coordinate.
- **price**: `float` - Sale price of the house.
- **bedrooms**: `integer` - Number of bedrooms.
- **bathrooms**: `float` - Number of bathrooms.
- **sqft_living**: `integer` - Square footage of living space.
- **sqft_lot**: `integer` - Square footage of the lot.
- **floors**: `float` - Number of floors.
- **waterfront**: `integer` - Waterfront view indicator.
- **view**: `integer` - Quality of the view.
- **condition**: `integer` - Condition of the house.
- **grade**: `integer` - Construction grade.
- **sqft_above**: `integer` - Square footage above ground.
- **yr_built**: `integer` - Year built.
- **yr_renovated**: `integer` - Year renovated.
- **sqft_living15**: `integer` - Living space of 15 nearest neighbors.
- **population**: `integer` - Population of the event's zip code.
- **high_schools**: `integer` - Number of high schools in the zip code.
- **middle_schools**: `integer` - Number of middle schools in the zip code.
- **primary_schools**: `integer` - Number of primary schools in the zip code.
- **other_schools**: `integer` - Number of other schools in the zip code.
- **unknown_schools**: `integer` - Number of unknown schools in the zip code.
- **total_schools**: `integer` - Total number of schools in the zip code.

### `cleaned_zipcode_populations`

- **zipcode**: `character(5)` - Primary key, zip code.
- **population**: `integer` - Population of the zip code.

### `cleaned_zipcode_public_schools`

- **zipcode**: `character(5)` - Primary key, zip code.
- **high_schools**: `integer` - Number of high schools.
- **middle_schools**: `integer` - Number of middle schools.
- **primary_schools**: `integer` - Number of primary schools.
- **other_schools**: `integer` - Number of other schools.
- **unknown_schools**: `integer` - Number of unknown schools.
- **total_schools**: `integer` - Total number of schools.

### `processed_event_ids`

- **id**: `bigint` - Primary key, ID of processed events.

## Notes

- Ensure that your PostgreSQL server is running and accessible from the environment where the script or Docker container is executed.
- Adjust the environment variables and Docker run command as necessary to fit your specific setup.
