"""Local Flask interface using a trusted model created by train.py."""
import json
import os
from pathlib import Path
import joblib
from flask import Flask, jsonify, request

ROOT = Path(__file__).resolve().parent

def create_app(model_path=None):
    app = Flask(__name__, static_folder=None)
    app.config['MAX_CONTENT_LENGTH'] = 100_000
    artifact = Path(model_path or os.environ.get('MODEL_PATH', ROOT / 'artifacts/classifier.joblib'))
    model = joblib.load(artifact) if artifact.is_file() else None
    dataset = 'unknown'
    metadata = artifact.with_name('metrics.json')
    if model is not None and metadata.is_file():
        dataset = json.loads(metadata.read_text(encoding='utf-8')).get('dataset', 'unknown')

    @app.get('/')
    def home():
        return (ROOT / 'index.html').read_text(encoding='utf-8')

    @app.get('/health')
    def health():
        return jsonify(status='ready' if model is not None else 'model_missing', dataset=dataset), 200 if model is not None else 503

    @app.post('/predict')
    def predict():
        data = request.get_json(silent=True)
        if not isinstance(data, dict) or not isinstance(data.get('resume'), str):
            return jsonify(error='Send a JSON object containing a resume string.'), 400
        text = data['resume'].strip()
        if not text or len(text) > 20000 or not any(c.isalpha() for c in text):
            return jsonify(error='Resume must contain letters and be between 1 and 20000 characters.'), 400
        if model is None:
            return jsonify(error='Model missing. Run python train.py first.'), 503
        if model.named_steps['tfidf'].transform([text]).nnz == 0:
            return jsonify(error='No recognized vocabulary. Try a more detailed summary.'), 422
        return jsonify(job_role=str(model.predict([text])[0]), dataset=dataset,
                       note='Synthetic demonstration model; not a hiring recommendation.' if dataset == 'synthetic_demo'
                       else 'Text category prediction; not an assessment of candidate suitability.')

    return app

if __name__ == '__main__':
    create_app().run(host='127.0.0.1', port=8080, debug=False)
