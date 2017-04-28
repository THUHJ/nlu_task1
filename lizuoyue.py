# ETH Zurich - Semester S17
# Natural Language Understanding - Task 1 - Part 1
# Team Member - Jie Huang, Yanping Xie, Zuoyue Li

# Import packages
from __future__ import print_function
import tensorflow as tf
import random
import numpy as np

# Deactivate the warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Set learning parameters
learning_rate  = 1e-2
training_iters = 5e5  
batch_size     = 64
display_step   = 10

# Set network parameters
n_vocab        = 100   # vocabulary size (shape of single sentence: 20k * 30)
n_steps        = 20    # sentence length
n_hidden       = 64    # dimension of hidden layer cell

# Create tf graph input
x = tf.placeholder(tf.int32, [batch_size, n_steps])
y = tf.placeholder(tf.float32, [batch_size, n_steps - 1, n_vocab])

# Define weights and bias
weight = {
	'i': tf.Variable(tf.random_normal([n_hidden, n_vocab])), \
	'h': tf.Variable(tf.random_normal([n_hidden, n_hidden])), \
	'o': tf.Variable(tf.random_normal([n_vocab, n_hidden])) \
}
bias = {
	'h': tf.Variable(tf.random_normal([n_hidden, 1])), \
	'o': tf.Variable(tf.random_normal([n_vocab, 1])) \
}

# Define RNN computation process
def RNN(x, weight, bias):
	x_one_hot = tf.one_hot(x, n_vocab)
	batch_inputs = tf.unstack(x_one_hot, axis = 0)
	rnn_outputs = []
	for batch_input in batch_inputs:
		rnn_inputs = tf.unstack(batch_input, axis = 0)
		state = bias['h']
		for rnn_input in rnn_inputs:
			state = tf.tanh(tf.add(tf.matmul(weight['i'], tf.reshape(rnn_input, shape = [n_vocab, 1])), tf.matmul(weight['h'], state)))
			rnn_outputs.append(tf.matmul(weight['o'], state))
		rnn_outputs.pop()
	return tf.reshape(rnn_outputs, shape = [batch_size, n_steps - 1, n_vocab])

pred = RNN(x, weight, bias)
print("Network Defined!")

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = pred, labels = y))
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 2), tf.argmax(y, 2))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables
init = tf.global_variables_initializer()

# Launch the graph
print("Start Training!")
with tf.Session() as sess:
	sess.run(init)
	step = 1
	# Keep training until reach max iterations
	while step * batch_size < training_iters:

		batch_x = np.zeros((batch_size, n_steps), dtype = np.int32)
		batch_y = np.zeros((batch_size, n_steps - 1, n_vocab))
		for i in range(batch_size):
			batch_x[i, 0] = random.randint(0, n_vocab - 1)
			for j in range(1, n_steps):
				batch_x[i, j] = (batch_x[i, j - 1] + 1) % n_vocab
				batch_y[i, j - 1, batch_x[i, j]] = 1
		# print(batch_x)
		# print(batch_y)
		# print("Data Done!")

		sess.run(optimizer, feed_dict = {x: batch_x, y: batch_y})
		# print("Optimize Done!")
		if step % display_step == 0:
			# Calculate batch accuracy
			acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
			# Calculate batch loss
			loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
			print(
				"Iter " + str(step * batch_size) + ", Minibatch Loss= " + \
				"{:.6f}".format(loss) + ", Training Accuracy= " + \
				"{:.5f}".format(acc) \
			)
		step += 1
	print("Optimization Finished!")

	# Calculate accuracy for test set
	# 
