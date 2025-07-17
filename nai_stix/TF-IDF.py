from elasticsearch import Elasticsearch
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def tfidf_vectorize(df):
    # Optional: clean pattern to remove structured parts like [ipv4-addr:value = '...']
    clean_pattern = df["pattern"].fillna("").str.replace(r"\[.*?\]", "", regex=True)

    text_data = clean_pattern + " " + df["labels"].fillna("") + " " + df["description"].fillna("")
    
    vectorizer = TfidfVectorizer(
        stop_words="english",       # remove common words like "the", "is"
        max_features=1000,          # limit to top 1000 words
        token_pattern=r"(?u)\b[a-zA-Z]{3,}\b"  # only words of length ≥3
    )
    vectors = vectorizer.fit_transform(text_data)
    
    return vectors, vectorizer.get_feature_names_out()

es = Elasticsearch("http://localhost:9200")

def fetch_stix_objects(index="stix_objects", size=1000):
    query = {"query": {"match_all": {}}}
    res = es.search(index=index, body=query, size=size)
    
    objects = []
    for hit in res['hits']['hits']:
        source = hit['_source']
        objects.append({
            "id": source.get("id"),
            "type": source.get("type"),
            "labels": ' '.join(source.get("labels", [])),
            "pattern": source.get("pattern", ""),
            "description": source.get("description", "")
        })
    return pd.DataFrame(objects)

if __name__ == "__main__":
    df = fetch_stix_objects()
    print("[+] Retrieved", len(df), "STIX objects")
    
    vectors, features = tfidf_vectorize(df)
    print("[+] TF-IDF shape:", vectors.shape)
    print("[+] Sample features:", features[:10])
