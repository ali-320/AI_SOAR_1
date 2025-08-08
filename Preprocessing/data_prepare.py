import os
import json
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from numpy.lib.format import open_memmap

def load_event_profiles(file_path):
    """Load event profiles from JSON."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def load_event_vectors(file_path):
    """Load event vectors from NumPy file."""
    return np.load(file_path)

def prepare_fnn_data(data, vector_file):
    """Prepare flat data for FNN: features and labels."""
    event_vectors = load_event_vectors(vector_file)
    labels = [entry['details']['evil'] for entry in tqdm(data, desc="Preparing FNN Data")]
    return event_vectors, np.array(labels)

def prepare_cnn_data(data, vector_file, sequence_length=10):
    """Prepare sequential data for CNN."""
    event_vectors = load_event_vectors(vector_file)
    sequences = []
    labels = []
    total_events = len(event_vectors)
    for i in tqdm(range(0, total_events - sequence_length + 1), desc="Preparing CNN Data"):
        seq = event_vectors[i:i+sequence_length]
        if len(seq) == sequence_length:
            sequences.append(seq)
            labels.append(data[i+sequence_length-1]['details']['evil'])
    return np.array(sequences), np.array(labels)

def estimate_lstm_total_sequences(data_len, sequence_length):
    return max(0, data_len - sequence_length + 1)

def prepare_lstm_data_mmap(data, vector_file, output_prefix, sequence_length=10, chunk_mb=2):
    """Prepare LSTM sequences in ~chunk_mb MB chunks and write to memory-mapped .npy files."""
    # Sort data by timestamp
    data_sorted = sorted(data, key=lambda x: float(x.get('timestamp', 0)))
    event_vectors = load_event_vectors(vector_file)

    # Map original positions
    id_map = {id(entry): i for i, entry in enumerate(data)}
    sorted_indices = [id_map[id(entry)] for entry in data_sorted]
    event_vectors_sorted = event_vectors[sorted_indices]

    # Estimate memory needs
    total_sequences = estimate_lstm_total_sequences(len(event_vectors_sorted), sequence_length)
    if total_sequences == 0:
        raise ValueError("Not enough data to form even one sequence.")

    feature_dim = event_vectors_sorted.shape[1]
    bytes_per_element = 8  # float64
    bytes_per_sequence = sequence_length * feature_dim * bytes_per_element
    sequences_per_chunk = max(1, (chunk_mb * 1024 * 1024) // bytes_per_sequence)

    # Create memory-mapped files
    X_mmap = open_memmap(f"{output_prefix}_X.npy", mode='w+', dtype='float64',
                         shape=(total_sequences, sequence_length, feature_dim))
    y_mmap = open_memmap(f"{output_prefix}_y.npy", mode='w+', dtype='bool',
                         shape=(total_sequences,))

    # Write data
    for i in tqdm(range(total_sequences), desc=f"Writing {output_prefix} (LSTM chunks)"):
        seq = event_vectors_sorted[i:i + sequence_length]
        label = data_sorted[i + sequence_length - 1].get('details', {}).get('evil', False)
        X_mmap[i] = seq
        y_mmap[i] = label

    print(f"✅ Saved: {output_prefix}_X.npy and {output_prefix}_y.npy")
    return X_mmap, y_mmap

def save_data(X, y, prefix):
    """Save prepared data to files."""
    np.save(f"{prefix}_X.npy", X)
    np.save(f"{prefix}_y.npy", y)

if __name__ == "__main__":
    # Load profiles
    print("Loading datasets...")
    train_data = load_event_profiles("./Preprocessing/Profiled/event_profiles_train.json")
    test_data = load_event_profiles("./Preprocessing/Profiled/event_profiles_test.json")
    val_data = load_event_profiles("./Preprocessing/Profiled/event_profiles_val.json")

    # Prepare FNN data
    print("Preparing FNN data...")
    X_fnn_train, y_fnn_train = prepare_fnn_data(train_data, "./Preprocessing/Vectorized_tfidf/event_vectors_train.npy")
    X_fnn_test, y_fnn_test = prepare_fnn_data(test_data, "./Preprocessing/Vectorized_tfidf/event_vectors_test.npy")
    X_fnn_val, y_fnn_val = prepare_fnn_data(val_data, "./Preprocessing/Vectorized_tfidf/event_vectors_val.npy")
    save_data(X_fnn_train, y_fnn_train, "./Preprocessing/DL_Data/FNN/fnn_train")
    save_data(X_fnn_test, y_fnn_test, "./Preprocessing/DL_Data/FNN/fnn_test")
    save_data(X_fnn_val, y_fnn_val, "./Preprocessing/DL_Data/FNN/fnn_val")

    # Prepare CNN data
    print("Preparing CNN data...")
    X_cnn_train, y_cnn_train = prepare_cnn_data(train_data, "./Preprocessing/Vectorized_tfidf/event_vectors_train.npy")
    X_cnn_test, y_cnn_test = prepare_cnn_data(test_data, "./Preprocessing/Vectorized_tfidf/event_vectors_test.npy")
    X_cnn_val, y_cnn_val = prepare_cnn_data(val_data, "./Preprocessing/Vectorized_tfidf/event_vectors_val.npy")
    save_data(X_cnn_train, y_cnn_train, "./Preprocessing/DL_Data/CNN/cnn_train")
    save_data(X_cnn_test, y_cnn_test, "./Preprocessing/DL_Data/CNN/cnn_test")
    save_data(X_cnn_val, y_cnn_val, "./Preprocessing/DL_Data/CNN/cnn_val")

    # Prepare LSTM data (with memory-mapped chunked writer)
    print("Preparing LSTM data...")
    prepare_lstm_data_mmap(train_data,
        "./Preprocessing/Vectorized_tfidf/event_vectors_train.npy",
        "./Preprocessing/DL_Data/LSTM/lstm_train")

    prepare_lstm_data_mmap(test_data,
        "./Preprocessing/Vectorized_tfidf/event_vectors_test.npy",
        "./Preprocessing/DL_Data/LSTM/lstm_test")

    prepare_lstm_data_mmap(val_data,
        "./Preprocessing/Vectorized_tfidf/event_vectors_val.npy",
        "./Preprocessing/DL_Data/LSTM/lstm_val")

    print("Data prepared for FNN, CNN, and LSTM for all datasets.")