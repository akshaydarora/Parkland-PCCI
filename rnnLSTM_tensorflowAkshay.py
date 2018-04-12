
# coding: utf-8

# # RNN- LSTM code [Classification Version]
:


# Import Essential Python Libraries

# Dependencies : Tensorflow , Pandas , NumPy and Sklearn

import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.python.framework import ops
from tensorflow.python.ops import clip_ops
from tensorflow.contrib.rnn import LSTMCell
from tensorflow.contrib.rnn.python.ops import core_rnn
from sklearn import preprocessing
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# Appreciate some randomness
tf.set_random_seed(1)

# Tuning the Hyperparameters

lr = 0.005    # learning rate
training_iters = 100000   # training epochs
batch_size = 50         # mini batch size
n_inputs = 30
n_steps = 6   # time steps
n_hidden_units = 256 # neurons in hidden layer
n_classes  = 2         # number of classes [recall , non-recall ]
num_layers =4     # number of layers in the hidden network
keep_prob  =0.8       # Dropout probability



# Read the dataset

df = pd.read_csv("/Users/brad/Documents/AkshayNew/sampleData/trainingData.dat")


data=df



dataset=np.array(data)
N = dataset.shape[0]
X_train = dataset[:,0:dataset.shape[1]-1]
# Targets have labels 1-indexed. We subtract one for 0-indexed
y_train = dataset[:,dataset.shape[1]-1]



y_train.shape




std_scale = preprocessing.StandardScaler().fit(X_train)
X_train = std_scale.transform(X_train)




X_train.shape




X_train=X_train.reshape(dataset.shape[0],n_steps,n_inputs)
X_train.shape


# Function for one hot encoding

def y2indicator(y):
    N = len(y)
    y = y.astype(np.int32)
    ind = np.zeros((N, 2))
    for i in range(N):
        ind[i, y[i]] = 1
    return ind

y_train=y2indicator(y_train)

df_test = pd.read_csv("/Users/brad/Documents/AkshayNew/sampleData/testingData.dat")


df_test.head()



data_test=df_test
testset=np.array(data_test)
M = testset.shape[0]
X_test = testset[:,0:testset.shape[1]-1]

# Targets have labels 1-indexed. We subtract one for 0-indexed
y_test = testset[:,testset.shape[1]-1]

y_test=y2indicator(y_test)



std_scale2 = preprocessing.StandardScaler().fit(X_test)
X_test = std_scale2.transform(X_test)




y_test.shape




X_test=X_test.reshape(testset.shape[0],n_steps,n_inputs)


X_test.shape
# One Hot Encoded : Targets  - Recall & Non - Recall
y_test

# tf Graph input

# Tensorflow Placeholders

x = tf.placeholder(tf.float32, [None, n_steps,n_inputs],name='Inputs')
y = tf.placeholder(tf.float32, [None,n_classes],name='Labels')

# Define weights and Biases

with tf.name_scope('Weights'):

    weights = {
    # (5,128)
    'in': tf.Variable(tf.truncated_normal([n_inputs, n_hidden_units]),name = 'W_in'),

    # (128, 2)
    'out': tf.Variable(tf.truncated_normal([n_hidden_units, n_classes]),name='W_out')

    }
    tf.summary.histogram('weights_in', weights['in'])
    tf.summary.histogram('weights_out', weights['out'])
with tf.name_scope('Biases'):

    biases = {
    # (128, )
    'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_units, ]),name='b_in'),
    # (2, )
    'out': tf.Variable(tf.constant(0.1, shape=[n_classes, ]),name='b_out')

    }
    tf.summary.histogram('biases_in', biases['in'])
    tf.summary.histogram('biases_out', biases['out'])

# Define a costumizable RNN function

def RNN(X, weights, biases):
    # hidden layer for input to cell

    X = tf.reshape(X, [-1, n_inputs])


    X_in = tf.matmul(X, weights['in']) + biases['in']
    X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_units])


    # basic LSTM Cell integrated with a Dropout Facility.

    cell=tf.contrib.rnn.DropoutWrapper(LSTMCell(n_hidden_units),output_keep_prob=keep_prob)

    # Feed single cell to Multi Dimensional RNN to add layers

    cell = tf.contrib.rnn.MultiRNNCell([cell]*num_layers)

    # lstm cell is divided into two parts (c_state, h_state)

    init_state = cell.zero_state(batch_size, dtype=tf.float32)
    # Use a dynamic RNN
    outputs, final_state = tf.nn.dynamic_rnn(cell, X_in, initial_state=init_state, time_major=False)

    outputs = tf.unstack(tf.transpose(outputs, [1,0,2]))

    results = tf.matmul(outputs[-1], weights['out']) + biases['out']    # shape = (128, 10)

    return results


# Function to load mini -batch operations

def next_batch(num, data, labels):

    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[ i] for i in idx]
    labels_shuffle = [labels[ i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)

# Calculating Loss function and its properties

with tf.name_scope("Softmax") as scope:
    with tf.variable_scope("Softmax_params"):

        beta = 0.01
        pred = RNN(x, weights, biases)
        # Cost Function
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y,name='logits'))
        # Loss function using L2 Regularization
        regularizer = tf.nn.l2_loss(weights['in'])
        cost = tf.reduce_mean(cost + beta * regularizer)
        correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        h1 = tf.summary.scalar('accuracy',accuracy)
        h2 = tf.summary.scalar('cost', cost)
        softmaxed_logits = tf.nn.softmax(pred)
        auc=tf.contrib.metrics.streaming_auc(
                              predictions=softmaxed_logits,
                              labels=y,
                              curve='ROC')
with tf.name_scope("Optimizer") as scope:
    train_op = tf.train.AdamOptimizer(lr).minimize(cost)


batch_xs, batch_ys = next_batch(batch_size,X_train,y_train)
batch_xs.shape

# Running the Neural Network with Tensorflow Session


with tf.Session() as sess:

    #Final code for the TensorBoard

    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter('logs/', sess.graph)
    init=tf.global_variables_initializer()
    sess.run(tf.local_variables_initializer())
    sess.run(init)
    step = 0
    while step * batch_size < training_iters:
        batch_xs, batch_ys = next_batch(batch_size,X_train,y_train)
        batch_xs = batch_xs.reshape([batch_size, n_steps, n_inputs])
        X_batch, y_batch = next_batch(batch_size,X_test,y_test)
        X_batch = X_batch.reshape([batch_size, n_steps, n_inputs])
        sess.run([train_op], feed_dict={
            x: batch_xs,
            y: batch_ys,
        })
        if step %50 == 0:
            cost_train,acc_train=sess.run([cost,accuracy], feed_dict={
            x: batch_xs,
            y: batch_ys,
            })

            #Evaluate validation performance [Testing Data]

            test_auc,cost_val,summ,acc_val = sess.run([auc,cost,merged,accuracy],feed_dict = {x: X_batch, y: y_batch})
            print('At %5.0f/%5.0f: Train COST %5.3f -- Test COST %5.3f -- Test Accuracy %5.3f ' %(step,training_iters,cost_train,cost_val,acc_val))


            #sess.run(tf.local_variables_initializer())
            #test_auc = sess.run(auc)

            print(test_auc)
            writer.add_summary(summ, step)
            writer.flush()

        step += 1

    print('Done !')   
