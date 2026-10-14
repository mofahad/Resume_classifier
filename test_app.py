import json
from pathlib import Path
import tempfile
import unittest
import joblib
from app import create_app
from model import make_pipeline
from train import load_rows

class ClassifierTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        path = Path(self.temp.name) / 'classifier.joblib'
        texts, labels = zip(*load_rows(None))
        joblib.dump(make_pipeline().fit(texts, labels), path)
        path.with_name('metrics.json').write_text(json.dumps({'dataset': 'synthetic_demo'}))
        self.client = create_app(path).test_client()

    def tearDown(self):
        self.temp.cleanup()

    def test_page_and_health(self):
        self.assertEqual(self.client.get('/').status_code, 200)
        self.assertEqual(self.client.get('/health').json['status'], 'ready')

    def test_prediction_and_provenance(self):
        response = self.client.post('/predict', json={'resume': 'Reconcile accounts and prepare tax returns and ledgers'})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json['job_role'], 'Accountant')
        self.assertEqual(response.json['dataset'], 'synthetic_demo')

    def test_invalid_inputs(self):
        for body in ({}, [], {'resume': None}, {'resume': '  '}, {'resume': 12}, {'resume': '1234'}, {'resume': 'a' * 50001}):
            self.assertEqual(self.client.post('/predict', json=body).status_code, 400)
        self.assertEqual(self.client.post('/predict', data='broken', content_type='application/json').status_code, 400)

    def test_unknown_vocabulary(self):
        self.assertEqual(self.client.post('/predict', json={'resume': 'zzqvvww'}).status_code, 422)

    def test_missing_model(self):
        client = create_app(Path(self.temp.name) / 'missing.joblib').test_client()
        self.assertEqual(client.get('/health').status_code, 503)
        self.assertEqual(client.post('/predict', json={'resume': 'Accounting experience'}).status_code, 503)

    def test_long_valid_resume_is_supported(self):
        text = ('Accounting ' * 4545) + 'books'
        self.assertEqual(len(text), 50000)
        self.assertEqual(self.client.post('/predict', json={'resume':text}).status_code, 200)

if __name__ == '__main__':
    unittest.main()
