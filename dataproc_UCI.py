import numpy as np
import pandas as pd
from collections import Counter

DATASET_PATH = "./UCI_data/raw/UCI HAR Dataset/"

INPUT_SIGNAL_TYPES = [
    "body_acc_x_",
    "body_acc_y_",
    "body_acc_z_",
    "body_gyro_x_",
    "body_gyro_y_",
    "body_gyro_z_",
    "total_acc_x_",
    "total_acc_y_",
    "total_acc_z_"
]


def load_x(X_signals_paths):
    X_signals = []

    for signal_type_path in X_signals_paths:
        with open(signal_type_path, "r") as f:
            X_signals.append(
                [np.array(serie, dtype=np.float32)
                 for serie in [row.replace('  ', ' ').strip().split(' ') for row in f]]
            )

    return np.transpose(X_signals, (1, 2, 0))


def load_y(y_path):
    # Read dataset from disk, dealing with text file's syntax
    with open(y_path, "r") as f:
        y = np.array(
            [elem for elem in [
                row.replace('  ', ' ').strip().split(' ') for row in f
            ]],
            dtype=np.int32
        )

    y = y.reshape(-1, )
    # Substract 1 to each output class for friendly 0-based indexing
    return y - 1


train_x_signals_paths = [
    DATASET_PATH + "train/Inertial Signals/" + signal + "train.txt" for signal in INPUT_SIGNAL_TYPES
]
test_x_signals_paths = [
    DATASET_PATH + "test/Inertial Signals/" + signal + "test.txt" for signal in INPUT_SIGNAL_TYPES
]

train_y_path = DATASET_PATH + "train/y_train.txt"
test_y_path = DATASET_PATH + "test/y_test.txt"

train_x = load_x(train_x_signals_paths)
test_x = load_x(test_x_signals_paths)

train_y = load_y(train_y_path)
test_y = load_y(test_y_path)
train_y_matrix = np.asarray(pd.get_dummies(train_y), dtype=np.int8)
test_y_matrix = np.asarray(pd.get_dummies(test_y), dtype=np.int8)

print(train_y, Counter(train_y))
print(test_y, Counter(test_y))

print(train_y_matrix)

np.save("./UCI_data/np/np_train_x.npy", train_x)
np.save("./UCI_data/np/np_train_y.npy", train_y_matrix)
np.save("./UCI_data/np/np_test_x.npy", test_x)
np.save("./UCI_data/np/np_test_y.npy", test_y_matrix)


# [4 4 4 ... 1 1 1] Counter({5: 1407, 4: 1374, 3: 1286, 0: 1226, 1: 1073, 2: 986})
# [4 4 4 ... 1 1 1] Counter({5: 537, 4: 532, 0: 496, 3: 491, 1: 471, 2: 420})
# [[0 0 0 0 1 0]
#  [0 0 0 0 1 0]
#  [0 0 0 0 1 0]
#  ...
#  [0 1 0 0 0 0]
#  [0 1 0 0 0 0]
#  [0 1 0 0 0 0]]

