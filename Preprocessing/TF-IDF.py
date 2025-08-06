import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler

# Global variables to reuse for test/val (fitted only on train)
vectorizer = None
scaler = None

def load_standardized_data(file_path):
    """Load the standardized JSON data from file."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def extract_textual_features(data):
    """Extract textual features from the data for TF-IDF processing."""
    documents = []
    for entry in data:
        text = f"{entry['threat_type']} {entry['details']['eventName']} {entry['details']['processName']}"
        args_text = " ".join([f"{arg.get('name', '')} {arg.get('value', '')}" for arg in entry['details']['args']])
        documents.append(f"{text} {args_text}".strip())
    return documents

def extract_numerical_features(data):
    """Extract numerical features like argsNum and returnValue."""
    numerical_features = []
    for entry in data:
        args_num = int(entry['details']['argsNum']) if entry['details']['argsNum'].isdigit() else 0
        return_value = int(entry['details']['returnValue']) if entry['details']['returnValue'].isdigit() else 0
        numerical_features.append([args_num, return_value])
    return np.array(numerical_features)

def apply_tfidf(documents, fit_vectorizer=False):
    """Apply TF-IDF to textual data, limiting to 500 features."""
    global vectorizer
    if fit_vectorizer and vectorizer is None:
        vectorizer = TfidfVectorizer(max_features=500, stop_words='english', lowercase=True)
        tfidf_matrix = vectorizer.fit_transform(documents)
        # Save feature names based on training data
        with open("./Preprocessing/Vectorized_tfidf/tfidf_feature_names.txt", "w") as f:
            f.write("\n".join(vectorizer.get_feature_names_out()))
    else:
        tfidf_matrix = vectorizer.transform(documents)  # Use pre-fitted vectorizer for test/val
    return tfidf_matrix.toarray()

def normalize_numerical_features(numerical_features, fit_scaler=False):
    """Normalize numerical features to a 0-1 range."""
    global scaler
    if fit_scaler and scaler is None:
        scaler = MinMaxScaler()
        normalized_features = scaler.fit_transform(numerical_features)
    else:
        normalized_features = scaler.transform(numerical_features)  # Use pre-fitted scaler for test/val
    return normalized_features, scaler

def combine_features(tfidf_array, normalized_numerical_features):
    """Combine TF-IDF vectors with normalized numerical features."""
    event_vectors = np.hstack((tfidf_array, normalized_numerical_features))
    return event_vectors

def save_event_vectors(event_vectors, output_file):
    """Save the event vectors to a NumPy file."""
    np.save(output_file, event_vectors)

def process_data(input_file, output_file, is_training=False):
    """Process the data through TF-IDF and event vector generation."""
    # Load data
    data = load_standardized_data(input_file)
    
    # Textual features for TF-IDF
    documents = extract_textual_features(data)
    tfidf_array = apply_tfidf(documents, fit_vectorizer=is_training)
    
    # Numerical features
    numerical_features = extract_numerical_features(data)
    normalized_numerical_features, scaler = normalize_numerical_features(numerical_features, fit_scaler=is_training)
    
    # Combine into event vectors
    event_vectors = combine_features(tfidf_array, normalized_numerical_features)
    save_event_vectors(event_vectors, output_file)
    print(f"Event vectors saved to {output_file}")

if __name__ == "__main__":
    # Process training data first to fit the vectorizer and scaler
    process_data("./Preprocessing/standardized_training_data.json", "./Preprocessing/Vectorized_tfidf/event_vectors_train.npy", is_training=True)
    # Process testing and validation data using the fitted vectorizer and scaler
    process_data("./Preprocessing/standardized_testing_data.json", "./Preprocessing/Vectorized_tfidf/event_vectors_test.npy", is_training=False)
    process_data("./Preprocessing/standardized_validation_data.json", "./Preprocessing/Vectorized_tfidf/event_vectors_val.npy", is_training=False)