import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout

def build_cnn(input_shape):
    """Build a 1D CNN model for sequence data."""
    model = Sequential([
        Conv1D(32, kernel_size=3, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

if __name__ == "__main__":
    # Load training and validation data
    X_train = np.load("./Preprocessing/DL_Data/CNN/cnn_train_X.npy")
    y_train = np.load("./Preprocessing/DL_Data/CNN/cnn_train_y.npy")
    X_val = np.load("./Preprocessing/DL_Data/CNN/cnn_val_X.npy")
    y_val = np.load("./Preprocessing/DL_Data/CNN/cnn_val_y.npy")

    # Build and train model
    model = build_cnn((X_train.shape[1], X_train.shape[2]))  # (sequence_length, num_features)
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))

    # Evaluate on test data
    X_test = np.load("./Preprocessing/DL_Data/CNN/cnn_test_X.npy")
    y_test = np.load("./Preprocessing/DL_Data/CNN/cnn_test_y.npy")
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"Test accuracy: {test_acc}")

    # Save model
    model.save("./Preprocessing/DL_Models/cnn_model.h5")
    print("CNN model trained and saved.")