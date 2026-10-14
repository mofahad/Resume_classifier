# Labeled resume baseline results

Recorded October 14, 2026. This is an initial text-category benchmark, not candidate ranking or a hiring-quality assessment.

## Data and split

The three supplied `Resume.csv` files are byte-identical. The labeled CSV contains 2,484 rows across 24 categories, including one missing resume. Cleaning removes that row and two exact duplicates after HTML removal, lowercasing, and whitespace normalization, leaving 2,481 rows.

The extracted and preprocessed CSVs lack category labels and are not used for supervised training. The supplied folder identifies this [Kaggle dataset reference](https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset); the recorded file hash, rather than a live dataset version, defines this run.

Source SHA-256: `816a7cd985a9a41e3ce8d1e83cb7e1beb1e854a89c1cf1fd0d75494faaa0559d`.

A stratified random split with seed 42 produces **1,861 training resumes and 620 test resumes**. TF-IDF is fitted only on training text. The saved pipeline applies identical preprocessing during serving. Aggregate split hashes and per-class metrics are in [the evaluation JSON](evaluation/labeled_baseline.json).

## Model comparison

| Model | Test accuracy | Macro F1 |
| --- | ---: | ---: |
| Training-set majority class | 4.84% | 0.0038 |
| TF-IDF unigrams/bigrams plus class-balanced logistic regression | 65.65% | 0.6303 |

The model correctly categorizes **407 of 620** test resumes. This evaluation did not tune hyperparameters against the test set. It is one split, not cross-validation or an external-source benchmark.

## Serving validation

The Flask test client reconstructs and verifies the recorded holdout hash before calling the actual prediction endpoint. All **620 resumes receive successful responses**, with the same 407 correct predictions. The original 20,000-character limit rejected five legitimate holdout resumes; the bounded limit is now 50,000 characters with a 250,000-byte request-body cap. The longest supplied resume is 38,842 characters.

Nine unit tests pass, covering API behavior, long valid input, missing models, unknown vocabulary, missing labels, normalized duplicate removal, and conflicting duplicate labels. API evaluation uses the same holdout as model evaluation; it is a serving consistency check, not a second independent benchmark.

## Limits

- Near-duplicate/template overlap is not measured or grouped before splitting.
- Categories are unevenly represented; several holdout classes have few examples.
- Source shifts, non-English resumes, short summaries, and fairness are not evaluated.
- No confidence calibration or suitability-for-employment assessment is provided.
- Raw resumes, identifiers, extracted text, and trained artifacts are not included in the repository changes.

The next modeling improvement should be selected with training-only validation, followed by an explicitly separate evaluation. Report failures and class-level results alongside the aggregate score.
