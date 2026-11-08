import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

def clean_text(text):
    return re.sub(r'\s+', ' ', re.sub(r'<[^>]*>', ' ', text)).strip().lower()

def make_pipeline():
    return Pipeline([
        ('tfidf', TfidfVectorizer(preprocessor=clean_text, ngram_range=(1, 2), max_features=10000)),
        ('classifier', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)),
    ])
