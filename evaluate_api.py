"""Check serving on the recorded holdout without exposing resume text."""
import argparse
import hashlib
import json
from pathlib import Path
from collections import Counter
from sklearn.model_selection import train_test_split
from app import create_app
from train import load_rows

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=Path, required=True)
    parser.add_argument('--model-dir', type=Path, required=True)
    args = parser.parse_args()
    metrics = json.loads((args.model_dir / 'metrics.json').read_text(encoding='utf-8'))
    rows = load_rows(args.csv)
    texts, labels = zip(*rows)
    _, holdout, _, truth = train_test_split(texts, labels, test_size=metrics['test_rows'], stratify=labels,
                                           random_state=metrics['seed'])
    split_hash = hashlib.sha256(json.dumps(list(zip(holdout, truth)), ensure_ascii=False,
                                           separators=(',', ':')).encode()).hexdigest()
    if split_hash != metrics['split']['test_sha256']:
        raise ValueError('Reconstructed holdout does not match the recorded split.')
    client = create_app(args.model_dir / 'classifier.joblib').test_client()
    statuses = Counter(); answered = correct = 0
    for text, label in zip(holdout, truth):
        response = client.post('/predict', json={'resume':text})
        statuses[str(response.status_code)] += 1
        if response.status_code == 200:
            answered += 1
            correct += response.json['job_role'] == label
    result = {'holdout_rows':len(holdout), 'answered_rows':answered, 'correct_answers':correct,
              'coverage':answered/len(holdout), 'accuracy_on_answered':correct/answered if answered else None,
              'status_counts':dict(statuses), 'dataset':metrics['dataset'],
              'test_sha256':split_hash,
              'limitations':'Same recorded holdout; not an additional independent evaluation. Full resumes differ from short summaries.'}
    (args.model_dir / 'api_evaluation.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
