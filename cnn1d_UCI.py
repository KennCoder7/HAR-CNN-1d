import numpy as np
import tensorflow as tf
from sklearn import metrics

# read data & labels
# labels = np.load('UCI_data/np/np_labels_2d_v1.npy')
# data = np.load('UCI_data/np/np_data_1d_v1.npy')
print("### Process1 --- data load ###")
# spilt
# spilt = np.random.rand(len(data)) < 0.7
train_x = np.load('UCI_data/np/np_train_x.npy')
train_y = np.load('UCI_data/np/np_train_y.npy')
test_x = np.load('UCI_data/np/np_test_x.npy')
test_y = np.load('UCI_data/np/np_test_y.npy')
print("### train_y (labels) shape: ", train_y.shape, " ###")
print("### Process2 --- data spilt ###")
# define
seg_len = 128
num_channels = 9
num_labels = 6
batch_size = 100
learning_rate = 0.001
num_epoches = 10000
num_batches = train_x.shape[0] // batch_size
print("### num_batch: ", num_batches, " ###")
training = tf.placeholder_with_default(False, shape=())
X = tf.placeholder(tf.float32, (None, seg_len, num_channels))
Y = tf.placeholder(tf.float32, (None, num_labels))
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
# loss_math = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=Y, logits=y_)
loss_math = Y * tf.log(tf.clip_by_value(y_, 1e-10, 1.0))
print("Y shape: ", Y.shape, "y_ shape:", y_.shape)
loss = -tf.reduce_mean(loss_math)
# print(xentropy.shape, loss.shape)
# define optimizer & training
opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = opt.minimize(loss)
# define accuracy
# correct = tf.nn.in_top_k(predictions=y_, targets=Y, k=1)
correct = tf.equal(tf.argmax(y_, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

# session
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    for epoch in range(num_epoches):
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
            cm = metrics.confusion_matrix(np.argmax(test_y, 1), pred_y,)
            print(cm, '\n')


# ### Epoch:  91 |Train loss =  0.0005881363 |Train acc =  0.96939605  ###
# ### Epoch:  92 |Train loss =  0.0006893406 |Train acc =  0.9654516  ###
# ### Epoch:  93 |Train loss =  0.00075024687 |Train acc =  0.94695324  ###
# ### Epoch:  94 |Train loss =  0.0011405542 |Train acc =  0.96354735  ###
# ### Epoch:  95 |Train loss =  0.0005345826 |Train acc =  0.968716  ###
# ### Epoch:  96 |Train loss =  0.0004642737 |Train acc =  0.97075623  ###
# ### Epoch:  97 |Train loss =  0.0006923626 |Train acc =  0.96803594  ###
# ### Epoch:  98 |Train loss =  0.0006373101 |Train acc =  0.9488574  ###
# ### Epoch:  99 |Train loss =  0.0006377074 |Train acc =  0.9678999  ###
# ### Epoch:  100 |Train loss =  0.0004101615 |Train acc =  0.97102827  ###
# ### After Epoch:  100  |Test acc =  0.90906006  ###
# [[470   6  20   0   0   0]
#  [ 39 403  29   0   0   0]
#  [  0   3 417   0   0   0]
#  [  4  13   0 400  74   0]
#  [  1   1   0  62 468   0]
#  [  1  26   0   0   0 510]]
#
# ### Epoch:  101 |Train loss =  0.00041779343 |Train acc =  0.9670838  ###
# ### Epoch:  102 |Train loss =  0.0003416327 |Train acc =  0.96735585  ###
# ### Epoch:  103 |Train loss =  0.00027735575 |Train acc =  0.9706202  ###
# ### Epoch:  104 |Train loss =  0.0007991481 |Train acc =  0.9664037  ###
# ### Epoch:  105 |Train loss =  0.0021402966 |Train acc =  0.96599567  ###
# ### Epoch:  106 |Train loss =  0.0006238035 |Train acc =  0.9585147  ###
# ### Epoch:  107 |Train loss =  0.00035862665 |Train acc =  0.9734766  ###
# ### Epoch:  108 |Train loss =  0.00048177902 |Train acc =  0.97320455  ###
# ### Epoch:  109 |Train loss =  0.0002757285 |Train acc =  0.9741567  ###
# ### Epoch:  110 |Train loss =  0.0007168199 |Train acc =  0.9495375  ###
# ### After Epoch:  110  |Test acc =  0.8876824  ###
# [[469  12  15   0   0   0]
#  [ 45 407  17   1   1   0]
#  [  2   4 414   0   0   0]
#  [  0  22   0 313 155   1]
#  [  1   1   0  23 507   0]
#  [  0  27   0   0   0 510]]

