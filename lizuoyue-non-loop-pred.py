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

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 2), tf.argmax(y, 2))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

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

f = open("../data/sentences.eval", 'r')

# Launch the graph
print("Start Predicting!")
with tf.Session() as sess:

	# Restore variables from disk.
	saver.restore(sess, "model.ckpt")
	print("Model restored.")

	for i in range(100):
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

		acc = sess.run(accuracy, feed_dict = {x: batch_x, y: batch_y})
		print(acc)
