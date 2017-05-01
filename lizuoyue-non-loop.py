# ETH Zurich - Semester S17
# Natural Language Understanding - Task 1 - Part 1
# Team Member - Jie Huang, Yanping Xie, Zuoyue Li

# Import packages
from __future__ import print_function
import tensorflow as tf
from tensorflow.contrib import rnn

import random
import numpy as np

# Deactivate the warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Set learning parameters
learning_rate  = 3e-3
training_iters = 1e5
batch_size     = 512
display_step   = 1

# Set network parameters
n_vocab        = 1000  # vocabulary size
n_steps        = 30    # sentence length
n_hidden       = 256   # dimension of hidden layer cell

# Create tf graph input
x = tf.placeholder(tf.int32, [batch_size, n_steps])
y = tf.placeholder(tf.float32, [batch_size, n_steps - 1, n_vocab])

# Define output weights and bias
weight = tf.Variable(tf.random_normal([n_vocab, n_hidden]))
bias = tf.Variable(tf.random_normal([n_vocab, 1]))

# Define RNN computation process 
def RNN(x, weight, bias):
	x_one_hot = tf.one_hot(x, n_vocab)
	inputs = tf.unstack(x_one_hot, axis = 0)
	lstm_cell = rnn.BasicLSTMCell(n_hidden)
	outputs, states = rnn.static_rnn(lstm_cell, inputs, dtype = tf.float32)
	final_outputs = [tf.transpose(tf.matmul(weight, tf.transpose(outputs[i][0: n_steps - 1, :])) + bias) for i in range(len(outputs))]
	return tf.reshape(final_outputs, shape = [batch_size, n_steps - 1, n_vocab])

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

# Add ops to save and restore all the variables
saver = tf.train.Saver()

# Construct vocabulary index dictionary
vocabulary = {}
f = open("vocabulary.txt", 'r')
line = f.readline()
idx = 0
while line:
	vocabulary[line.strip()] = idx;
	idx += 1
	line  = f.readline()
f.close()

f = open("../data/sentences.train", 'r')

# Launch the graph
print("Start Training!")
with tf.Session() as sess:
	sess.run(init)
	step = 1
	# Keep training until reach max iterations
	while step * batch_size < training_iters:

		batch_x = []
		while len(batch_x) < batch_size:
			line = f.readline()
			if not line:
				f.close()
				f = open("../data/sentences.train", 'r')
				line = f.readline()

			words = line.strip().split(' ')
			if (len(words) <= 28):
				code = [vocabulary["<bos>"]]
				for word in words:
					if word in vocabulary:
						code.append(vocabulary[word])
					else:
						code.append(vocabulary["<unk>"])
				while (len(code) < 29):
					code.append(vocabulary["<pad>"])
				code.append(vocabulary["<eos>"])
				batch_x.append(code)

		batch_x = np.array(batch_x)
		# batch_x = np.zeros((batch_size, n_steps), dtype = np.int32)
		batch_y = np.zeros((batch_size, n_steps - 1, n_vocab))
		for i in range(batch_size):
			# batch_x[i, 0] = random.randint(0, n_vocab - 1)
			for j in range(1, n_steps):
				# batch_x[i, j] = (batch_x[i, j - 1] + 1) % n_vocab
				batch_y[i, j - 1, batch_x[i, j]] = 1
		# print(batch_x)
		# print(batch_y)
		# print("Data Done!")

		sess.run(optimizer, feed_dict = {x: batch_x, y: batch_y})
		# print("Optimize Done!")
		if step % display_step == 0:
			# Calculate batch accuracy
			acc = sess.run(accuracy, feed_dict = {x: batch_x, y: batch_y})
			# Calculate batch loss
			loss = sess.run(cost, feed_dict = {x: batch_x, y: batch_y})
			print(
				"Iter " + str(step * batch_size) + ", Minibatch Loss = " + \
				"{:.6f}".format(loss) + ", Training Accuracy = " + \
				"{:.6f}".format(acc) \
			)
		step += 1
	print("Optimization Finished!")

	save_path = saver.save(sess, "model.ckpt")
	print("Model saved in file: %s" % save_path)
