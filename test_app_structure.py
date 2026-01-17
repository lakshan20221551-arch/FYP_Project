import unittest
from app import app

class BasicTests(unittest.TestCase):

    def setUp(self):
        app.config['TESTING'] = True
        app.config['WTF_CSRF_ENABLED'] = False
        self.app = app.test_client()

    def test_main_page(self):
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Student Success Prediction', response.data)
        self.assertIn(b'Marital Status', response.data)

    def test_predict_route_missing_model(self):
        # Since we haven't trained the model (no data.csv), the app should gracefully handle missing artifacts
        # based on our implementation logic in app.py
        response = self.app.post('/predict', data={
            'Marital status': '1',
            'Nacionality': '1',
            'Displaced': '1',
            'Gender': '1',
            'Age at enrollment': '20'
            # simplified data payload
        })
        self.assertEqual(response.status_code, 200)
        # We expect the error message from app.py
        self.assertIn(b'Error: Model not loaded', response.data)

if __name__ == "__main__":
    unittest.main()
