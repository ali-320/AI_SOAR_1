import numpy as np
import tensorflow as tf

batch_size = 32

def make_ds(X, y, shuffle=False):
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        ds = ds.shuffle(buffer_size=16384, reshuffle_each_iteration=True)
    # cast on the fly to avoid loading everything into RAM as float64
    ds = ds.map(lambda a, b: (tf.cast(a, tf.float32), tf.cast(b, tf.float32)),
                num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

# load fnn model
model_fnn = tf.keras.models.load_model("./Preprocessing/DL_Models/fnn_model.h5")

# load test data
X_test_fnn = np.load("./Preprocessing/DL_Data/FNN/fnn_test_X.npy")
y_test_fnn = np.load("./Preprocessing/DL_Data/FNN/fnn_test_y.npy")

test_loss_fnn, test_acc_fnn = model_fnn.evaluate(X_test_fnn, y_test_fnn)
print(f"\nTest accuracy of fnn model: {test_acc_fnn}\n")

# load cnn model
model_cnn = tf.keras.models.load_model("./Preprocessing/DL_Models/cnn_model.h5")

# load test data
X_test_cnn = np.load("./Preprocessing/DL_Data/CNN/cnn_test_X.npy")
y_test_cnn = np.load("./Preprocessing/DL_Data/CNN/cnn_test_y.npy")

test_loss_cnn, test_acc_cnn = model_cnn.evaluate(X_test_cnn, y_test_cnn)
print(f"\nTest accuracy of cnn model: {test_acc_cnn}\n")

# load LSTM model
model_lstm = tf.keras.models.load_model("./Preprocessing/DL_Models/lstm_model_final.h5")

# load test data
X_test = np.load("./Preprocessing/DL_Data/LSTM/lstm_test_X.npy")
y_test = np.load("./Preprocessing/DL_Data/LSTM/lstm_test_y.npy")

test_ds  = make_ds(X_test,  y_test)

test_loss_lstm, test_acc_lstm = model_lstm.evaluate(test_ds)
print(f"\nTest accuracy of lstm model: {test_acc_lstm}\n")