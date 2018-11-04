import numpy as np

# labels = np.load('ACT_data/np/np_real_str_labels_1d_v1.npy')
# data = np.load('ACT_data/np/np_data_1d.npy')
#
# labels = np.load('ACT_data/np/np_real_str_labels_1d_v1.npy')
# print(labels)
# print(np.unique(labels))
# print(labels.shape[0])
# from collections import Counter
# print(Counter(labels))

# ['Jogging' 'Jogging' 'Jogging' ... 'Sitting' 'Sitting' 'Sitting']
# ['Downstairs' 'Jogging' 'Sitting' 'Standing' 'Upstairs' 'Walking']
# 24403
# Counter({'Walking': 9429, 'Jogging': 7605, 'Upstairs': 2733, 'Downstairs': 2228, 'Sitting': 1332, 'Standing': 1076})

# a = np.array(([1, 2, 3], [4, 5, 6]), dtype=int)
# print(a)
# for i in a:
#     print(i)
#     print(i.shape)
#
# print(i for i in a)

a = np.arange(270).reshape(1, 90, 3)
print(a.shape)
b = a.transpose()
print(b.shape)
