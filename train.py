"""Train a baseline. Synthetic demo examples are not a performance benchmark."""
import argparse
import csv
import json
from pathlib import Path
import joblib
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from model import clean_text, make_pipeline

ROOT = Path(__file__).resolve().parent
DEMO = {
    'Data Scientist': [
        'Analyze datasets with Python SQL regression and hypothesis testing.',
        'Build statistical models and report findings using pandas and Tableau.',
        'Evaluate classification models using cross validation and feature engineering.',
        'Study time series forecast demand and measure statistical uncertainty.',
        'Investigate sampling bias using regression and exploratory data analysis.',
        'Design experiments calculate metrics and communicate analytical results.',
    ],
    'Software Engineer': [
        'Develop backend REST APIs using Python Flask and SQL databases.',
        'Implement Java services write unit tests and debug application code.',
        'Build React interfaces and integrate frontend applications with APIs.',
        'Design software architecture review code and maintain CI pipelines.',
        'Implement authentication database migrations and application services.',
        'Develop reusable software components and test application behavior.',
    ],
    'Accountant': [
        'Reconcile accounts prepare financial statements and maintain ledgers.',
        'Manage invoices accounts payable and accounts receivable.',
        'Prepare tax returns review financial records and audit transactions.',
        'Perform month end closing journal entries and budget reconciliation.',
        'Maintain balance sheets income statements and accounting records.',
        'Process payroll reconcile bank statements and review expenses.',
    ],
}

def load_rows(path):
    if path is None:
        return [(text, label) for label, examples in DEMO.items() for text in examples]
    with open(path, encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        if not {'Resume_str', 'Category'} <= set(reader.fieldnames or []):
            raise ValueError('CSV must contain Resume_str and Category columns.')
        rows = [(r['Resume_str'], r['Category'].strip()) for r in reader
                if r['Resume_str'] and r['Category'] and clean_text(r['Resume_str'])]
    unique = {}
    for text, label in rows:
        key = clean_text(text)
        if key in unique and unique[key][1] != label:
            raise ValueError('Identical normalized text has conflicting categories.')
        unique[key] = (text, label)
    return list(unique.values())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=Path)
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts')
    args = parser.parse_args()
    rows = load_rows(args.csv)
    counts = {}
    for _, label in rows:
        counts[label] = counts.get(label, 0) + 1
    if len(counts) < 2 or min(counts.values(), default=0) < 4:
        parser.error('Supply at least two categories with four unique examples each.')
    texts, labels = zip(*rows)
    x_train, x_test, y_train, y_test = train_test_split(
        texts, labels, test_size=max(len(counts), round(len(rows) * .25)), stratify=labels, random_state=42)
    model = make_pipeline().fit(x_train, y_train)
    predictions = model.predict(x_test)
    metrics = {
        'dataset': 'user_supplied_csv' if args.csv else 'synthetic_demo',
        'seed': 42, 'train_rows': len(x_train), 'test_rows': len(x_test), 'class_counts': counts,
        'accuracy': accuracy_score(y_test, predictions),
        'macro_f1': f1_score(y_test, predictions, average='macro', zero_division=0),
        'classification_report': classification_report(y_test, predictions, output_dict=True, zero_division=0),
        'limitations': 'Synthetic demo scores do not estimate real resume performance.' if not args.csv
                      else 'One stratified holdout; near-duplicates and source bias are not measured.',
    }
    args.output.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.output / 'classifier.joblib')
    (args.output / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in metrics.items() if k != 'classification_report'}, indent=2))

if __name__ == '__main__':
    main()
