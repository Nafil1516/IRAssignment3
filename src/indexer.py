import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

def build_tfidf_index(docs, index_dir):
    os.makedirs(index_dir, exist_ok=True)
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(docs)

    # Save index
    with open(os.path.join(index_dir, 'vectorizer.pkl'), 'wb') as f:
        pickle.dump(vectorizer, f)
    with open(os.path.join(index_dir, 'tfidf_matrix.pkl'), 'wb') as f:
        pickle.dump(tfidf_matrix, f)
    return vectorizer, tfidf_matrix

def load_tfidf_index(index_dir):
    try:
        with open(os.path.join(index_dir, 'vectorizer.pkl'), 'rb') as f:
            vectorizer = pickle.load(f)
        with open(os.path.join(index_dir, 'tfidf_matrix.pkl'), 'rb') as f:
            tfidf_matrix = pickle.load(f)
        return vectorizer, tfidf_matrix
    except Exception:
        return None, None
