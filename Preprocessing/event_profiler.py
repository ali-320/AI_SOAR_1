import json
import numpy as np
from sklearn.cluster import KMeans

def load_event_vectors(file_path):
    """Load event vectors from a NumPy file."""
    return np.load(file_path)

def load_standardized_data(file_path):
    """Load the standardized JSON data."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def apply_clustering(event_vectors, n_clusters=5):
    """Cluster event vectors using KMeans."""
    #randome_state makes the random selection of vectors consistent. If this is not defined, then I will get different result every time I run the code.
    #42 is selected because it is the answer to the ultimate question of life, the universe, and everything according to Douglas Adams' "The Hitchhiker's Guide to the Galaxy".
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(event_vectors)
    return clusters, kmeans

def save_event_profiles(standardized_data, clusters, output_file):
    """Add cluster labels to the data and save as event profiles."""
    for i, entry in enumerate(standardized_data):
        entry['profile'] = int(clusters[i])  # Add profile as an integer
    with open(output_file, 'w') as f:
        json.dump(standardized_data, f, indent=2)

def profile_events(vector_file, standardized_file, output_file):
    """Profile events by clustering their vectors."""
    # Load data
    event_vectors = load_event_vectors(vector_file)
    standardized_data = load_standardized_data(standardized_file)
    
    # Cluster events
    clusters, kmeans = apply_clustering(event_vectors)
    
    # Save profiles
    save_event_profiles(standardized_data, clusters, output_file)
    print(f"Event profiles saved to {output_file}")

if __name__ == "__main__":
    # Profile training, testing, and validation data
    profile_events("./Preprocessing/Vectorized_tfidf/event_vectors_train.npy", "./Preprocessing/standardized_training_data.json", "./Preprocessing/Profiled/event_profiles_train.json")
    profile_events("./Preprocessing/Vectorized_tfidf/event_vectors_test.npy", "./Preprocessing/standardized_testing_data.json", "./Preprocessing/Profiled/event_profiles_test.json")
    profile_events("./Preprocessing/Vectorized_tfidf/event_vectors_val.npy", "./Preprocessing/standardized_validation_data.json", "./Preprocessing/Profiled/event_profiles_val.json")