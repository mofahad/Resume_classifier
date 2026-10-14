# Resume text classifier

A local text-categorization prototype using TF-IDF and class-balanced logistic regression. Train a baseline, inspect holdout metrics, and try predictions through Flask. No paid services or API keys are required.

## Run locally

Use Python 3.11 or 3.12:

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python train.py
python app.py
```

Open http://127.0.0.1:8080. The default model uses **18 synthetic examples across three categories** to demonstrate the pipeline. It is not a validated resume classifier. The API identifies the dataset used for training.

## Train with labeled data

```bash
python train.py --csv /path/to/Resume.csv
```

The UTF-8 CSV must have `Resume_str` and `Category` columns, at least two categories, and at least four unique examples per category. Supply data you are authorized to use; no resume dataset is included. Exact normalized duplicates are removed before a reproducible stratified holdout split. Conflicting labels for identical text stop training. Near-duplicate leakage is not yet addressed.

TF-IDF is fitted on training data only. Preprocessing lives inside the saved pipeline, so serving uses the same transformations as training. Outputs are `artifacts/classifier.joblib` and `artifacts/metrics.json`, containing accuracy, macro F1, per-class results, split counts, provenance, and limitations. Synthetic-demo results do not establish real-world accuracy.

## Recorded labeled-data evaluation

See [BASELINE_RESULTS.md](BASELINE_RESULTS.md) for the supplied 24-category CSV run: data cleaning, split hashes, majority-class comparison, per-class metrics, and API coverage. The CSV and trained model remain local; aggregate results are in `evaluation/labeled_baseline.json`.

```powershell
python train.py --csv C:\path\to\Resume.csv --output artifacts/dataset-baseline
python evaluate_api.py --csv C:\path\to\Resume.csv --model-dir artifacts/dataset-baseline
$env:MODEL_PATH="artifacts/dataset-baseline/classifier.joblib"
python app.py
```

Keep the same file content and row order to reproduce the recorded split. Source and split hashes are recorded in the metrics. Training does not use the unlabeled extracted/preprocessed CSVs.

## API and checks

`GET /health` returns 200 when the model is available and 503 when missing. `POST /predict` accepts `{"resume": "Your resume text or summary"}` and returns a `job_role`, dataset provenance, and a usage note. Input is limited to 50,000 characters and a 250,000-byte request body. Invalid input returns 400; oversized bodies return 413; text without known vocabulary returns 422.

```bash
python -m unittest -v
```

Tests cover the page, readiness, inference, invalid requests, long valid resumes, missing artifacts, unknown vocabulary, deduplication, conflicting labels, and missing category columns. GitHub Actions runs tests and demo training. Full-dataset training requires the separately supplied CSV.

## Limitations and original work

This prototype categorizes text; it does not rank candidates or determine suitability for employment. Its single labeled-data holdout does not establish performance across other resume sources, short summaries, languages, or demographic groups. It has no fairness validation or production deployment. Requests are processed locally without application-level storage. Only load trusted model artifacts, because joblib deserialization can execute code.

Original notebooks remain as historical experiments. Malformed duplicate notebooks have been restored from their valid counterparts. The repaired command-line path replaces missing model files, the incorrect template directory, and inconsistent preprocessing. Historical notebooks are not the supported execution path.
