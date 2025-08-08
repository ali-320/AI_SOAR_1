#To check the shape of the .npy files for the training of all three models
#It is mentioned in logs.txt at Test #12.
import numpy as np
print(np.load("./Preprocessing/DL_Data/FNN/fnn_train_X.npy").shape) #output: (number_of_samples, number_of_features)
print(np.load("./Preprocessing/DL_Data/CNN/cnn_train_X.npy").shape) #output: (num_of_seq, seq_length, num_of_features)
print(np.load("./Preprocessing/DL_Data/LSTM/lstm_train_X.npy").shape) #output: (num_of_seq, seq_length, num_of_features)