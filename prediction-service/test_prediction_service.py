import unittest
import os
import json
import requests
import time
from datetime import datetime, timedelta

class PredictionServiceTests(unittest.TestCase):
    def setUp(self):
        # Get host and port from environment variables or use defaults
        self.host = os.environ.get('TEST_HOST', 'localhost')
        self.port = os.environ.get('TEST_PORT', '8888')
        self.base_url = f'http://{self.host}:{self.port}'
        self.endpoint = f'{self.base_url}/predicted-home-value'
        
        # Wait for service to be ready
        self._wait_for_service()
    
    def _wait_for_service(self, max_retries=5, retry_delay=2):
        """Wait for the service to be available"""
        for i in range(max_retries):
            try:
                response = requests.get(self.base_url)
                # Even if we get a 404, the service is running
                return
            except requests.ConnectionError:
                if i < max_retries - 1:
                    time.sleep(retry_delay)
                else:
                    self.fail("Service is not available")
    
    def test_valid_request(self):
        """Test a valid request with all required fields"""
        data = {
            "property": {
                "sale_date": datetime.now().strftime('%Y-%m-%d'),
                "sqft_living": 2000,
                "sqft_lot": 4000,
                "sqft_above": 1500,
                "sqft_basement": 500,
                "sqft_living15": 2000,
                "sqft_lot15": 4000,
                "year_built": 1975,
                "year_renovated": None,
                "zipcode": "98001",  # Use a valid zipcode from your database
                "latitude": 47.3,
                "longitude": -122.2,
                "floors": 1.5,
                "waterfront": False,
                "bedrooms": 3,
                "bathrooms": 2.5
            }
        }
        
        response = requests.post(
            self.endpoint,
            json=data,
            headers={"Content-Type": "application/json"}
        )
        
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertIn("predicted_price", result)
        self.assertIsInstance(result["predicted_price"], float)
        
    def test_missing_required_field(self):
        """Test a request with a missing required field"""
        data = {
            "property": {
                # Missing sqft_living which is required
                "sale_date": datetime.now().strftime('%Y-%m-%d'),
                "sqft_lot": 4000,
                "sqft_above": 1500,
                "zipcode": "98001",
                "latitude": 47.3,
                "longitude": -122.2
            }
        }
        
        response = requests.post(
            self.endpoint,
            json=data,
            headers={"Content-Type": "application/json"}
        )
        
        self.assertEqual(response.status_code, 400)
        result = response.json()
        self.assertIn("error", result)
        
    def test_invalid_field_values(self):
        """Test a request with invalid field values"""
        data = {
            "property": {
                "sale_date": datetime.now().strftime('%Y-%m-%d'),
                "sqft_living": -100,  # Invalid: negative value
                "sqft_lot": 4000,
                "sqft_above": 1500,
                "zipcode": "98001",
                "latitude": 47.3,
                "longitude": -122.2
            }
        }
        
        response = requests.post(
            self.endpoint,
            json=data,
            headers={"Content-Type": "application/json"}
        )
        
        self.assertEqual(response.status_code, 400)
        result = response.json()
        self.assertIn("error", result)
        
    def test_cleanable_data(self):
        """Test a request with messy but cleanable data"""
        data = {
            "property": {
                "sale_date": datetime.now().strftime('%Y-%m-%d'),
                "sqft_living": "2000",  # String instead of int
                "sqft_lot": 4000,
                "sqft_above": 1500,
                "zipcode": "98001",
                "latitude": 47.3,
                "longitude": -122.2,
                "waterfront": "true"  # String instead of boolean
            }
        }
        
        response = requests.post(
            self.endpoint,
            json=data,
            headers={"Content-Type": "application/json"}
        )
        
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertIn("predicted_price", result)
        
    def test_invalid_zipcode(self):
        """Test a request with a zipcode that doesn't exist in the enrichment database"""
        data = {
            "property": {
                "sale_date": datetime.now().strftime('%Y-%m-%d'),
                "sqft_living": 2000,
                "sqft_lot": 4000,
                "sqft_above": 1500,
                "zipcode": "00000",  # Invalid zipcode
                "latitude": 47.3,
                "longitude": -122.2
            }
        }
        
        response = requests.post(
            self.endpoint,
            json=data,
            headers={"Content-Type": "application/json"}
        )
        
        self.assertEqual(response.status_code, 400)
        result = response.json()
        self.assertIn("error", result)
        
    def test_missing_property_field(self):
        """Test a request missing the 'property' field"""
        data = {
            "not_property": {
                "sale_date": datetime.now().strftime('%Y-%m-%d'),
                "sqft_living": 2000,
                "sqft_lot": 4000,
                "sqft_above": 1500,
                "zipcode": "98001",
                "latitude": 47.3,
                "longitude": -122.2
            }
        }
        
        response = requests.post(
            self.endpoint,
            json=data,
            headers={"Content-Type": "application/json"}
        )
        
        self.assertEqual(response.status_code, 400)
        result = response.json()
        self.assertIn("error", result)

if __name__ == '__main__':
    unittest.main()