import tensorflow as tf
import numpy as np
import random
import read_m_data
import matplotlib.pyplot as plt

batch_size = 100
step = 20000
Learn_rate = 0.001
dataset_size = read_m_data.n_train
Dimension = read_m_data.d_sample
P_train = read_m_data.P_train
T_train = read_m_data.T_train
P_valid = read_m_data.P_valid
T_valid = read_m_data.T_valid
T_test_num = read_m_data.n_test

def add_layer(inputs, in_size, out_size,  n_layer, activation_function=None):
    layer_name = 'layer%s' % n_layer
    global Weights
    Weights = tf.Variable(tf.random_normal([in_size, out_size]), name='W')
    tf.summary.histogram(layer_name + '/weights', Weights)
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, name='b')
    tf.summary.histogram(layer_name + '/biases', biases)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    Wx_plus_b = tf.nn.dropout(Wx_plus_b, keep_prob)
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    tf.summary.histogram(layer_name + '/outputs', outputs)
    return outputs

keep_prob = tf.placeholder(tf.float32)
with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32, [None, Dimension], name='P_train')
    ys = tf.placeholder(tf.float32, [None, 1], name='T_train')

l1 = add_layer(xs, Dimension, 50,  n_layer = 1, activation_function=tf.nn.relu)
#l2 = add_layer(l1, 50, 10,  n_layer = 2,activation_function=tf.nn.relu)
#l3 = add_layer(l2, 10, 30,  n_layer = 3, activation_function=tf.nn.relu)
prediction = add_layer(l1, 50, 1,  n_layer = 2, activation_function=None)
sess = tf.Session()

loss = tf.reduce_mean(tf.square(ys - prediction))
tf.summary.scalar('loss', loss)

accuracy = 1 - tf.reduce_mean(tf.abs((ys - prediction)/ys))
tf.summary.scalar('accuracy', accuracy)
R_square = 1 - tf.reduce_sum(tf.square(prediction - ys)) / tf.reduce_sum(tf.square(ys - tf.reduce_mean(ys)))
tf.summary.scalar('R2', R_square)

with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer (Learn_rate).minimize(loss)#

merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter("logs/train", sess.graph)
test_writer = tf.summary.FileWriter("logs/valid", sess.graph)

init = tf.global_variables_initializer()
saver = tf.train.Saver()
sess.run(init)

plt.xlim([1,step])
plt.ion()
xx = [0,0]
yy_train = [1,1]
yy_test = [1,1]

for i in range(step):
    start = (i*batch_size) % dataset_size
    end = min(start + batch_size, dataset_size)
    sess.run(train_step, feed_dict={xs: P_train[start:end], ys: T_train[start:end], keep_prob: 1})
    if i % 50 == 0:
        print(i,\
            sess.run(loss, feed_dict={xs: P_train, ys: T_train, keep_prob: 1}),\
            sess.run(R_square, feed_dict={xs: P_train, ys: T_train, keep_prob: 1}),\
            sess.run(R_square, feed_dict={xs: P_valid, ys: T_valid, keep_prob: 1}),\
            sess.run(accuracy, feed_dict={xs: P_train, ys: T_train, keep_prob: 1}),\
            sess.run(accuracy, feed_dict={xs: P_valid, ys: T_valid, keep_prob: 1}))
        xx[0] = xx[1]
        yy_train[0] = yy_train[1]
        yy_test[0] = yy_test[1]
        xx[1] = i
        yy_train[1] = sess.run(R_square, feed_dict={xs: P_train, ys: T_train, keep_prob: 1})
        yy_test[1] = sess.run(R_square, feed_dict={xs: P_valid, ys: T_valid, keep_prob: 1})
        #plt.semilogy(xx, yy_train)
        #plt.semilogy(xx, yy_test)
        plt.plot(xx, yy_train)
        plt.plot(xx, yy_test)
        plt.pause(0.1)
        train_result = sess.run(merged, feed_dict={xs: P_train, ys: T_train, keep_prob: 1})
        test_result = sess.run(merged, feed_dict={xs: P_valid, ys: T_valid, keep_prob: 1})
        train_writer.add_summary(train_result, i)
        test_writer.add_summary(test_result, i)
    
P_valid,T_sim = sess.run([xs,prediction],feed_dict={xs: P_valid, keep_prob: 1})
T_valid = np.array(T_valid).reshape([1,T_test_num])[0]
T_sim = np.array(T_sim).reshape([1,T_test_num])[0]

RMAE = max(np.fabs(T_valid - T_sim))/np.std(T_valid,ddof=1)
ACCURACY = np.mean(1 - np.true_divide(np.fabs(np.subtract(T_sim,T_valid)),T_valid))
R_square = 1 - sum(np.square((T_sim-T_valid)))/sum(np.square((T_valid-np.mean(T_valid))))
RAAE = np.mean(np.fabs(np.subtract(T_sim,T_valid)))/np.std(T_valid,ddof=1)

P_train,T_prediction = sess.run([xs,prediction],feed_dict={xs: P_train, keep_prob: 1})
T_train = np.array(T_train).reshape([1,T_test_num])[0]
T_prediction = np.array(T_prediction).reshape([1,T_test_num])[0]

train_RMAE = max(np.fabs(T_train - T_prediction))/np.std(T_train,ddof=1)
train_ACCURACY = np.mean(1 - np.true_divide(np.fabs(np.subtract(T_prediction,T_train)),T_train))
train_R_square = 1 - sum(np.square((T_prediction-T_train)))/sum(np.square((T_train-np.mean(T_train))))
train_RAAE = np.mean(np.fabs(np.subtract(T_prediction,T_train)))/np.std(T_train,ddof=1)
#print('ACCURACY:%s' % ACCURACY)
#print('R_square:%s' % R_square)
#print('RMAE:%s' % RMAE)
#print('RAAE:%s' % RAAE)
print('测试数据')
print('ACCURACY','\t','R_square','\t','RMAE','\t','RAAE')
print(ACCURACY,'\t',R_square,'\t',RMAE,'\t',RAAE)
print('训练数据')
#print('ACCURACY','\t','R_square','\t','RMAE','\t','RAAE')
print(train_ACCURACY,'\t',train_R_square,'\t',train_RMAE,'\t',train_RAAE)
