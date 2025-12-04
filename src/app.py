# src/app.py
import os
import pandas as pd
import streamlit as st
from indexer import build_tfidf_index, load_tfidf_index
from retriever import search_tfidf_cosine
from preprocessing import clean_text, normalize_text, tokenize


DATA_FILE = r"C:\Users\Lenovo\OneDrive\Desktop\IR_ASSIGNMENT\data\Articles.csv"
INDEX_DIR = os.path.join('..','indexes')


@st.cache_data
def load_data(path):
    encodings = ['utf-8', 'cp1252', 'latin-1']
    df = None
    for enc in encodings:
        try:
            df = pd.read_csv(path, encoding=enc, on_bad_lines='skip')
            print(f"Loaded CSV with encoding: {enc}")
            break
        except Exception:
            continue
    if df is None:
        st.error("Could not read CSV with common encodings.")
        return None

    if 'Heading' not in df.columns:
        df['Heading'] = df.index.astype(str)
    if 'Article' not in df.columns:
        df['Article'] = ''
    df['Article'] = df['Article'].fillna('')
    return df


@st.cache_data
def preprocess_documents(df):
    processed_docs = []
    for doc in df['Article']:
        tokens = tokenize(normalize_text(clean_text(doc)))
        processed_docs.append(' '.join(tokens))
    return processed_docs


@st.cache_data
def ensure_index(processed_docs):
    vec, mat = load_tfidf_index(INDEX_DIR)
    if vec is None or mat is None:
        vec, mat = build_tfidf_index(processed_docs, INDEX_DIR)
    return vec, mat

def main():
    st.set_page_config(page_title="Document Search", layout="wide")
    st.title(" Document Search System")

    df = load_data(DATA_FILE)
    if df is None:
        return

    processed_docs = preprocess_documents(df)
    vectorizer, tfidf_matrix = ensure_index(processed_docs)

    query = st.text_input("Enter your search query:")
    top_k = st.slider("Number of results", min_value=1, max_value=10, value=5)

    if st.button("Search") and query.strip():
        q_proc = ' '.join(tokenize(normalize_text(clean_text(query))))
        results, _ = search_tfidf_cosine(q_proc, vectorizer, tfidf_matrix, df, top_k=top_k)

        if results:
            st.success(f"Retrieved {len(results)} results")
            for r in results:
                st.markdown(f"**Rank {r['rank']} | Score: {r['score']}**")
                st.markdown(f"**Heading:** {r['heading']}")
                st.markdown(f"**Snippet:** {r['snippet']}")
                st.markdown("---")
        else:
            st.warning("No results found.")

if __name__ == "__main__":
    main()
