import numpy as np
import tensorflow as tf
from sklearn import metrics

# read data & labels
labels = np.load('ADL_data/np/np_labels_1d_v1.npy')
data = np.load('ADL_data/np/np_data_1d_v1.npy')
print("### Process1 --- data load ###")
# spilt
spilt = np.random.rand(len(data)) < 0.7
train_x = data[spilt]
train_y = labels[spilt]
test_x = data[~spilt]
test_y = labels[~spilt]
print("### train_y shape: ", train_y.shape, " ###")
print("### Process2 --- data spilt ###")
# define
seg_len = 128
num_channels = 3
num_labels = 7
batch_size = 100
learning_rate = 0.001
num_epochs = 200
num_batches = train_x.shape[0] // batch_size
print("### num_batch: ", num_batches, " ###")
training = tf.placeholder_with_default(False, shape=())
X = tf.placeholder(tf.float32, (None, seg_len, num_channels))
Y = tf.placeholder(tf.int32, (None))
print("### Process3 --- define ###")

# CNN
# convolution layer 1
conv1 = tf.layers.conv1d(
    inputs=X,
    filters=32,
    kernel_size=2,
    strides=1,
    padding='same',
    activation=tf.nn.relu
)
print("### convolution layer 1 shape: ", conv1.shape, " ###")

# pooling layer 1
pool1 = tf.layers.max_pooling1d(
    inputs=conv1,
    pool_size=4,
    strides=2,
    padding='same'
)
print("### pooling layer 1 shape: ", pool1.shape, " ###")

# convolution layer 2
conv2 = tf.layers.conv1d(
    inputs=pool1,
    filters=64,
    kernel_size=2,
    strides=1,
    padding='same',
    activation=tf.nn.relu
)
print("### convolution layer 2 shape: ", conv2.shape, " ###")

# pooling layer 2
pool2 = tf.layers.max_pooling1d(
    inputs=conv2,
    pool_size=4,
    strides=2,
    padding='same'
)
print("### pooling layer 2 shape: ", pool2.shape, " ###")

# convolution layer 3
conv3 = tf.layers.conv1d(
    inputs=pool2,
    filters=128,
    kernel_size=2,
    strides=1,
    padding='same',
    activation=tf.nn.relu
)
print("### convolution layer 3 shape: ", conv3.shape, " ###")

# pooling layer 3
pool3 = tf.layers.max_pooling1d(
    inputs=conv3,
    pool_size=4,
    strides=2,
    padding='same'
)
print("### pooling layer 3 shape: ", pool3.shape, " ###")

# flat output
l_op = pool3
shape = l_op.get_shape().as_list()
flat = tf.reshape(l_op, [-1, shape[1] * shape[2]])
print("### flat shape: ", flat.shape, " ###")

# fully connected layer 1
fc1 = tf.layers.dense(
    inputs=flat,
    units=100,
    activation=tf.nn.tanh
)
fc1 = tf.nn.dropout(fc1, keep_prob=0.5)
print("### fully connected layer 1 shape: ", fc1.shape, " ###")
# bn_fc1 = tf.layers.batch_normalization(fc1, training=training)
# bn_fc1_act = tf.nn.relu(bn_fc1)

# fully connected layer 1
fc2 = tf.layers.dense(
    inputs=fc1,
    units=100,
    activation=tf.nn.tanh
)
fc2 = tf.nn.dropout(fc2, keep_prob=0.5)
print("### fully connected layer 2 shape: ", fc2.shape, " ###")
# bn_fc2 = tf.layers.batch_normalization(fc2, training=training)
# bn_fc2_act = tf.nn.relu(bn_fc2)

# fully connected layer 3
fc3 = tf.layers.dense(
    inputs=fc2,
    units=num_labels,
    activation=tf.nn.softmax
)
print("### fully connected layer 3 shape: ", fc3.shape, " ###")

# prediction
# y_ = tf.layers.batch_normalization(fc3, training=training)
y_ = fc3
print("### prediction shape: ", y_.get_shape(), " ###")

# define loss
loss_math = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=Y, logits=y_)
# loss_math = Y * tf.log(tf.clip_by_value(y_, 1e-10, 1.0))
print("Y shape: ", Y.shape, "y_ shape:", y_.shape)
loss = tf.reduce_mean(loss_math)
# print(xentropy.shape, loss.shape)
# define optimizer & training
opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = opt.minimize(loss)
# define accuracy
correct = tf.nn.in_top_k(predictions=y_, targets=Y, k=1)
# correct = tf.equal(tf.argmax(y_, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

# session
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    for epoch in range(num_epochs):
        for i in range(num_batches):
            offset = (i * batch_size) % (train_y.shape[0] - batch_size)
            batch_x = train_x[offset:(offset + batch_size)]
            batch_y = train_y[offset:(offset + batch_size)]
            _, c = sess.run([train_op, loss], feed_dict={X: batch_x, Y: batch_y, training: True})
        print("### Epoch: ", epoch+1, "|Train loss = ", c,
              "|Train acc = ", sess.run(accuracy, feed_dict={X: train_x, Y: train_y}), " ###")
        if (epoch+1) % 10 == 0:
            print("### After Epoch: ", epoch+1,
                  " |Test acc = ", sess.run(accuracy, feed_dict={X: test_x, Y: test_y}), " ###")
            pred_y = sess.run(tf.argmax(y_, 1), feed_dict={X: test_x})
            cm = metrics.confusion_matrix(y_true=test_y, y_pred=pred_y)
            print(cm, '\n')


# 2018/11/2
# ### Epoch:  200 |Train loss =  1.1654272 |Train acc =  0.82973075  ###
# ### After Epoch:  200  |Test acc =  0.74527454  ###
# [[119   1   0   1   0   5   5]
#  [  1   0   4   0   2   1 135]
#  [  0   0  70   2   4   0   9]
#  [  0   0   0 110   7  14   0]
#  [  4   0   2   8 108  13  31]
#  [  1   0   1  11   5 115   2]
#  [  1   0   0   3   7   1 308]]

# Y = tf.placeholder(tf.int32, (None)) --> Y = tf.placeholder(tf.float32, (None))
# loss_math = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=Y, logits=y_)
# --> loss_math = Y * tf.log(tf.clip_by_value(y_, 1e-10, 1.0))
# correct = tf.nn.in_top_k(predictions=y_, targets=Y, k=1)
# --> correct = tf.equal(tf.argmax(y_, 1), tf.argmax(Y, 1))
# 数据格式不兼容
