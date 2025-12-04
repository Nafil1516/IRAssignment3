# src/retriever.py
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from preprocessing import clean_text, normalize_text, tokenize, snippet

def search_tfidf_cosine(query, vectorizer, tfidf_matrix, df, top_k=5):
    if not query.strip():
        return [], 0.0

    query_tokens = tokenize(normalize_text(clean_text(query)))
    query_processed = " ".join(query_tokens)
    query_vector = vectorizer.transform([query_processed])
    scores = cosine_similarity(query_vector, tfidf_matrix).flatten()
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        if scores[idx] > 0:
            results.append({
                'rank': len(results) + 1,
                'score': scores[idx],
                'heading': df.iloc[idx]['Heading'],
                'snippet': snippet(df.iloc[idx]['Article'])
            })
    return results, 0  
