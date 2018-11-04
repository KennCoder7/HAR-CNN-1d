import numpy as np
import tensorflow as tf
from sklearn import metrics

# read data & labels
labels = np.load('ACT_data/np/np_labels_1d.npy')
data = np.load('ACT_data/np/np_data_1d.npy')
print("### Process1 --- data load ###")
# spilt
spilt = np.random.rand(len(data)) < 0.7
train_x = data[spilt]
train_y = labels[spilt]
test_x = data[~spilt]
test_y = labels[~spilt]
print("### Process2 --- data spilt ###")
# define
seg_len = 90
num_channels = 3
num_labels = 6
batch_size = 200
learning_rate = 0.0001
num_epochs = 1000
num_batches = train_x.shape[0] // batch_size
print("### num_batch: ", num_batches, " ###")
X = tf.placeholder(tf.float32, (None, seg_len, num_channels))
Y = tf.placeholder(tf.int32, (None))
print("### Process3 --- define ###")

# CNN
# convolution layer 1
conv1 = tf.layers.conv1d(
    inputs=X,
    filters=60,
    kernel_size=60,
    strides=1,
    padding='valid',
    activation=tf.nn.relu
)
print("### convolution layer 1 shape: ", conv1.shape, " ###")
# pooling layer 1
pool1 = tf.layers.max_pooling1d(
    inputs=conv1,
    pool_size=20,
    strides=2,
    padding='valid'
)
print("### pooling layer 1 shape: ", pool1.shape, " ###")
# convolution layer 2
conv2 = tf.layers.conv1d(
    inputs=pool1,
    filters=180,
    kernel_size=6,
    strides=1,
    padding='valid',
    activation=tf.nn.relu
)
print("### convolution layer 2 shape: ", conv2.shape, " ###")
# flat output
l_op = conv2
shape = l_op.get_shape().as_list()
flat = tf.reshape(l_op, [-1, shape[1] * shape[2]])
print("### flat shape: ", flat.shape, " ###")
# fully connected layer 1
fc1 = tf.layers.dense(
    inputs=flat,
    units=100,
    activation=tf.nn.relu
)
fc1 = tf.nn.dropout(fc1, keep_prob=0.8)
print("### fully connected layer 1 shape: ", fc1.shape, " ###")
# fully connected layer 2
fc2 = tf.layers.dense(
    inputs=fc1,
    units=100,
    activation=tf.nn.relu
)
fc2 = tf.nn.dropout(fc2, keep_prob=0.8)
print("### fully connected layer 2 shape: ", fc2.shape, " ###")
# fully connected layer 3
fc3 = tf.layers.dense(
    inputs=fc2,
    units=num_labels,
    activation=tf.nn.softmax
)
print("### fully connected layer 3 shape: ", fc3.shape, " ###")
# prediction
y_ = fc3
print("### prediction shape: ", y_.get_shape(), " ###")

# define loss
xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=Y, logits=y_)
loss = tf.reduce_mean(xentropy)
# print(xentropy.shape, loss.shape)
# define optimizer & training
opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = opt.minimize(loss)
# define accuracy
correct = tf.nn.in_top_k(predictions=y_, targets=Y, k=1)
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

# session
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    for epoch in range(num_epochs):
        for i in range(num_batches):
            offset = (i * batch_size) % (train_y.shape[0] - batch_size)
            batch_x = train_x[offset:(offset + batch_size)]
            batch_y = train_y[offset:(offset + batch_size)]
            _, c = sess.run([train_op, loss], feed_dict={X: batch_x, Y: batch_y})
        if (epoch + 1) % 10 == 0:
            print("### Epoch: ", epoch+1, "|Train loss = ", c,
                  "|Train acc = ", sess.run(accuracy, feed_dict={X: train_x, Y: train_y}), " ###")
        if (epoch + 1) % 50 == 0:
            print("### After Epoch: ", epoch+1,
                  " |Test acc = ", sess.run(accuracy, feed_dict={X: test_x, Y: test_y}), " ###")
        if (epoch + 1) % 100 == 0:
            pred_y = sess.run(tf.argmax(y_, 1), feed_dict={X: test_x})
            cm = metrics.confusion_matrix(y_true=test_y, y_pred=pred_y)
            print(cm, '\n')

# 2018/11/2  try1 (3->180->1080  no drop  1fc)
# Epoch:  76 |Train loss =  1.064422 |Train acc =  0.97551835  ###
# Epoch:  77 |Train loss =  1.0664105 |Train acc =  0.9766897  ###
# Epoch:  78 |Train loss =  1.0663525 |Train acc =  0.97633827  ###
# Epoch:  79 |Train loss =  1.0665829 |Train acc =  0.96339464  ###
# Epoch:  80 |Train loss =  1.0666713 |Train acc =  0.9786225  ###
# |Test acc =  0.95661074  ###

# 2018/11/2 latest version (MORE faster trainning)
# ### Epoch:  141 |Train loss =  1.058779 |Train acc =  0.9836056  ###
# ### Epoch:  142 |Train loss =  1.058729 |Train acc =  0.9836056  ###
# ### Epoch:  143 |Train loss =  1.0588478 |Train acc =  0.98366374  ###
# ### Epoch:  144 |Train loss =  1.059034 |Train acc =  0.98372185  ###
# ### Epoch:  145 |Train loss =  1.0589324 |Train acc =  0.98372185  ###
# ### After Epoch:  145  |Test acc =  0.95292974  ###
# [[ 565   29    2    0   52   25]
#  [  14 2197    0    0   14   20]
#  [   2    0  349   21    1    0]
#  [   1    0    0  317    1    0]
#  [  52   34    4    1  691   19]
#  [  24    6    0    0   16 2745]]

# 2018/11/4 15:07 fc1 fc2 tanh-->relu
# ### Epoch:  950 |Train loss =  1.0435917 |Train acc =  0.98663163  ###
# ### After Epoch:  950  |Test acc =  0.9571016  ###