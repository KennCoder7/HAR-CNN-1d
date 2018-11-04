import pandas as pd
import numpy as np
from scipy import stats


def read_data(file_path):
    column_names = ['user-id', 'activity', 'timestamp', 'x-axis', 'y-axis', 'z-axis']
    data = pd.read_csv(file_path, header=None, names=column_names)
    return data  # return a Pandas data frame


def feature_normalize(dt):  # normalise each of the accelerometer component (i.e. x, y and z)
    mu = np.mean(dt, axis=0)
    sigma = np.std(dt, axis=0)
    return (dt - mu) / sigma


print("### Process1 --- Read data --- Started ###")
dataset = read_data('ADL_data/raw/adl_raw.txt')
print("### Check nan of dataset before dropna(): ###")
print(dataset.isnull().any())
dataset = dataset.dropna()
print("### Check nan of dataset after dropna(): ###")
print(dataset.isnull().any())
print("### Process1 --- Read data --- Finished ###")

print("### Process2 --- Normalization --- Started ###")
dataset['x-axis'] = round(dataset['x-axis'], 2)
dataset['y-axis'] = round(dataset['y-axis'], 2)
dataset['z-axis'] = round(dataset['z-axis'], 2)
dataset['x-axis'] = feature_normalize(dataset['x-axis'])
dataset['y-axis'] = feature_normalize(dataset['y-axis'])
dataset['z-axis'] = feature_normalize(dataset['z-axis'])
print("### Check nan of dataset_normalize before dropna(): ###")
print(dataset.isnull().any())
dataset = dataset.dropna()
print("### Check nan of dataset_normalize after dropna(): ###")
print(dataset.isnull().any())
# print(dataset)
print("### Process2 --- Normalization --- Finished ###")


def windows(data, size):
    start = 0
    while start < data.count():
        yield start, start + size
        start += (size / 2)


def segment_signal(data, window_size=128):
    segments = np.empty((0, window_size, 3))
    labels = np.empty((0))
    n = 0
    for (start, end) in windows(data["timestamp"], window_size):
        start = int(start)
        end = int(end)
        x = data["x-axis"][start:end]
        y = data["y-axis"][start:end]
        z = data["z-axis"][start:end]
        if len(dataset["timestamp"][start:end]) == window_size:
            segments = np.vstack([segments, np.dstack([x, y, z])])
            labels = np.append(labels, stats.mode(data["activity"][start:end])[0][0])
        if start-n > 0.1*data["timestamp"].count():
            n = start
            print("### Process3 --- Segment --- In progress --- [ ",
                  100*round(start/data['timestamp'].count(), 2), "% ] Finished ###")
    return segments, labels


def str_to_num(list):
    list_unique = np.unique(list)
    list_unique_len = len(list_unique)
    new_list = np.array([], dtype=int)
    for i in list:
        for j in range(list_unique_len):
            if list_unique[j] == i:
                new_list = np.append(new_list, int(j))
    return new_list


print("### Process3 --- Segment --- Started ###")
segments, labels = segment_signal(dataset)
print("### Process3 --- Segment --- Finished ###")

print("### Process4 --- Labels Transform --- Started ###")
unique_labels = np.unique(labels)
num_labels = str_to_num(labels)
unique_num_labels = np.unique(num_labels)
unique_labels = np.append(unique_labels, unique_num_labels)
d2_labels = np.asarray(pd.get_dummies(labels), dtype=np.int8)
print("### Process4 --- Labels Transform --- Finished ###")

print("### Process5 --- Save --- Started ###")
np.save('ADL_data/np/np_data_1d_v1.npy', segments)
np.save('ADL_data/np/np_labels_1d_v1.npy', num_labels)
print("num_labels: ", num_labels)
np.save('ADL_data/np/np_real_str_labels_1d_v1.npy', labels)
print("real_labels: ", labels)
np.save('ADL_data/np/np_unique_labels_1d_v1.npy', unique_labels)
print("unique_labels: ", unique_labels)
print("labels number: ", labels.shape[0])
np.save('ADL_data/np/np_labels_2d_v1.npy', d2_labels)
print("2d labels: ", d2_labels)
print("### Process5 --- Save --- Finished ###")

# 2018/11/2
# ### Process1 --- Read data --- Started ###
# ### Check nan of dataset before dropna(): ###
# user-id      False
# activity     False
# timestamp    False
# x-axis       False
# y-axis       False
# z-axis       False
# dtype: bool
# ### Check nan of dataset after dropna(): ###
# user-id      False
# activity     False
# timestamp    False
# x-axis       False
# y-axis       False
# z-axis       False
# dtype: bool
# ### Process1 --- Read data --- Finished ###
# ### Process2 --- Normalization --- Started ###
# ### Check nan of dataset_normalize before dropna(): ###
# user-id      False
# activity     False
# timestamp    False
# x-axis       False
# y-axis       False
# z-axis       False
# dtype: bool
# ### Check nan of dataset_normalize after dropna(): ###
# user-id      False
# D:\Anaconda\lib\site-packages\scipy\stats\stats.py:245: RuntimeWarning:
# The input array could not be properly checked for nan values. nan values will be ignored.
# activity     False
#   "values. nan values will be ignored.", RuntimeWarning)
# timestamp    False
# x-axis       False
# y-axis       False
# z-axis       False
# dtype: bool
# ### Process2 --- Normalization --- Finished ###
# ### Process3 --- Segment --- Started ###
# ### Process3 --- Segment --- In progress --- [  10.0 % ] Finished ###
# ### Process3 --- Segment --- In progress --- [  20.0 % ] Finished ###
# ### Process3 --- Segment --- In progress --- [  30.0 % ] Finished ###
# ### Process3 --- Segment --- In progress --- [  40.0 % ] Finished ###
# ### Process3 --- Segment --- In progress --- [  50.0 % ] Finished ###
# ### Process3 --- Segment --- In progress --- [  60.0 % ] Finished ###
# ### Process3 --- Segment --- In progress --- [  70.0 % ] Finished ###
# ### Process3 --- Segment --- In progress --- [  80.0 % ] Finished ###
# ### Process3 --- Segment --- In progress --- [  90.0 % ] Finished ###
# ### Process3 --- Segment --- Finished ###
# ### Process4 --- Labels Transform --- Started ###
# ### Process4 --- Labels Transform --- Finished ###
# ### Process5 --- Save --- Started ###
# num_labels:  [0 0 0 ... 6 6 6]
# real_labels:  ['Brush_teeth' 'Brush_teeth' 'Brush_teeth' ... 'Walk' 'Walk' 'Walk']
# unique_labels:  ['Brush_teeth' 'Climb_stairs' 'Comb_hair' 'Drink_glass' 'Getup_bed'
#  'Pour_water' 'Walk' '0' '1' '2' '3' '4' '5' '6']
# labels number:  3748
# 2d labels:  [[1 0 0 ... 0 0 0]
#  [1 0 0 ... 0 0 0]
#  [1 0 0 ... 0 0 0]
#  ...
#  [0 0 0 ... 0 0 1]
#  [0 0 0 ... 0 0 1]
#  [0 0 0 ... 0 0 1]]
# ### Process5 --- Save --- Finished ###

