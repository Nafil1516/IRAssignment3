import os
import pandas as pd
from indexer import build_tfidf_index, load_tfidf_index
from retriever import search_tfidf_cosine
from preprocessing import clean_text, normalize_text, tokenize

DATA_FILE = r"C:\Users\Lenovo\OneDrive\Desktop\IR_ASSIGNMENT\data\Articles.csv"
INDEX_DIR = os.path.join('..','indexes')

def load_data(path):
    encodings = ['utf-8', 'cp1252', 'latin-1']
    df = None
    for enc in encodings:
        try:
            df = pd.read_csv(path, encoding=enc, on_bad_lines='skip')
            print(f"Loaded CSV with encoding: {enc}")
            break
        except Exception as e:
            print(f"Failed with encoding {enc}: {e}")
            continue
    if df is None:
        print("Could not read CSV with common encodings.")
        return None

    if 'Heading' not in df.columns:
        df['Heading'] = df.index.astype(str)
    if 'Article' not in df.columns:
        df['Article'] = ''
    df['Article'] = df['Article'].fillna('')
    return df

def preprocess_documents(df):
    processed_docs = []
    for doc in df['Article']:
        tokens = tokenize(normalize_text(clean_text(doc)))
        processed_docs.append(' '.join(tokens))
    return processed_docs

def ensure_index(processed_docs):
    vec, mat = load_tfidf_index(INDEX_DIR)
    if vec is None or mat is None:
        print("Building TF-IDF index...")
        vec, mat = build_tfidf_index(processed_docs, INDEX_DIR)
        print("Index built.")
    else:
        print("Index loaded.")
    return vec, mat

def interactive_loop(df, vectorizer, tfidf_matrix):
    print(f"\n--- Interactive search over {len(df)} documents ---")
    print("Type 'exit' or 'quit' to stop.")
    while True:
        try:
            q = input("\nEnter query: ").strip()
            if q.lower() in ('exit','quit'):
                print("Goodbye.")
                break
            if not q:
                continue
            results, _ = search_tfidf_cosine(q, vectorizer, tfidf_matrix, df, top_k=5)
            print(f"\nRetrieved {len(results)} results")
            if not results:
                print("No results found.")
            for r in results:
                print('-'*50)
                print(f"Rank: {r['rank']} | Score: {r['score']:.4f}")
                print(f"Heading: {r['heading']}")
                print(f"Snippet: {r['snippet']}")
            print('-'*50)
        except KeyboardInterrupt:
            print("\nInterrupted, exiting.")
            break
        except Exception as e:
            print("Error during search:", e)
            break

def main():
    print("Loading data from:", DATA_FILE)
    df = load_data(DATA_FILE)
    if df is None:
        return
    processed_docs = preprocess_documents(df)
    vectorizer, tfidf_matrix = ensure_index(processed_docs)
    interactive_loop(df, vectorizer, tfidf_matrix)

if __name__ == '__main__':
    main()
