# ETH Zurich, Semester S17
# Natural Language Understanding, Task 1
# Team Members: Jie Huang, Yanping Xie, Zuoyue Li

from __future__ import print_function

# Deactivate the warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Import packages
import tensorflow as tf
import numpy as np
import random

print("Import packages ... Done!")

# Set learning parameters
learning_rate  = 2e-1  # learning rate
training_iters = 1e5   # training iters
global_norm    = 10.0  # global norm
disp_step      = 1     # display step

# Set network parameters
batch_size     = 64    # batch size
vocab_size     = 20000 # vocabulary size
emb_size       = 100   # word embedding size
seq_length     = 30    # sequence length
state_size     = 1024  # hidden state size
keep_prob      = 1.0   # for dropout wrapper
forget_bias    = 1.0   # for cell
proj_size      = 512

# Define RNN network input and output
x = tf.placeholder(tf.int32, [None, seq_length    ])
y = tf.placeholder(tf.int32, [None, seq_length - 1])
y_one_col = tf.reshape(y, [-1])
y_one_hot = tf.reshape(tf.one_hot(y, vocab_size), [-1, vocab_size])

# Define word embeddings, output weight and output bias
emb_weight = tf.get_variable("emb_weight", [vocab_size, emb_size  ], dtype = tf.float32, initializer = tf.contrib.layers.xavier_initializer())
proj_weight = tf.get_variable("proj_weight", [state_size, proj_size  ], dtype = tf.float32, initializer = tf.contrib.layers.xavier_initializer())
out_weight = tf.get_variable("out_weight", [proj_size, vocab_size], dtype = tf.float32, initializer = tf.contrib.layers.xavier_initializer())
out_bias   = tf.get_variable("out_bias"  , [vocab_size]            , dtype = tf.float32, initializer = tf.contrib.layers.xavier_initializer())

# Define LSTM cell weights and biases
with tf.variable_scope("basic_lstm_cell"):
	weights = tf.get_variable("weights", [emb_size + state_size, 4 * state_size], dtype = tf.float32, initializer = tf.contrib.layers.xavier_initializer())
	biases  = tf.get_variable("biases" , [4 * state_size], dtype = tf.float32, initializer = tf.contrib.layers.xavier_initializer())

print("Define network parameters ... Done!")

# Define RNN computation process
input_emb   = tf.nn.embedding_lookup(emb_weight, x)
input_seq   = tf.unstack(input_emb, axis = 1)
lstm_cell   = tf.contrib.rnn.BasicLSTMCell(state_size, reuse = True, forget_bias = forget_bias)
# lstm_cell   = tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob = keep_prob)
state       = lstm_cell.zero_state(batch_size, tf.float32)
output_seq  = []
for input_unit in input_seq:
	output_unit, state = lstm_cell(input_unit, state)
	output_seq.append(output_unit)
last_state  = state
output_seq  = tf.transpose(output_seq[0: len(output_seq) - 1], [1, 0, 2])
output_seq  = tf.reshape(output_seq, [-1, state_size])
pred_logits = tf.matmul(tf.matmul(output_seq, proj_weight), out_weight) + out_bias

print("Define network computation process ... Done!")

# Define loss and optimizer
loss      = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = pred_logits, labels = y_one_col))
opt_func  = tf.train.AdamOptimizer(learning_rate = learning_rate)
grad, var = zip(*opt_func.compute_gradients(loss))
grad, _   = tf.clip_by_global_norm(grad, global_norm)
optimizer = opt_func.apply_gradients(zip(grad, var))

# Evaluate model
true_pred = tf.equal(tf.argmax(pred_logits, 1), tf.to_int64(y_one_col))
accuracy  = tf.reduce_mean(tf.cast(true_pred, tf.float32))

# Initialize the variables
init      = tf.global_variables_initializer()
saver     = tf.train.Saver()

print("Define loss, optimizer and evaluate function ... Done!")

# Construct vocabulary index dictionary
vocabulary = {}
look_up = []
f = open("vocabulary.txt", 'r')
line = f.readline()
idx = 0
while line:
	look_up.append(line.strip())
	vocabulary[look_up[idx]] = idx;
	idx += 1
	line  = f.readline()
f.close()

print("Load dictionary ... Done!")

# Launch the graph
print("Start Training!")
f = open("../data/sentences.train", 'r')
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
			if len(words) <= seq_length - 2:
				code = [vocabulary["<bos>"]]
				for word in words:
					if word in vocabulary:
						code.append(vocabulary[word])
					else:
						code.append(vocabulary["<unk>"])
				while len(code) <= seq_length - 2:
					code.append(vocabulary["<pad>"])
				code.append(vocabulary["<eos>"])
				batch_x.append(code)

		# Random generation of input data
		"""
		batch_x = []
		for k in range(batch_size):
			code = [random.randint(0, vocab_size - 1)]
			for i in range(seq_length - 1):
				code.append((code[i] + 1) % vocab_size)
			batch_x.append(code)
		"""
		batch_x = np.array(batch_x)
		batch_y = batch_x[:, 1: seq_length]

		if step > 0: # == 1:
			feed_dict = {x: batch_x, y: batch_y}
		else:
			feed_dict = {x: batch_x, y: batch_y, state: state_feed}

		sess.run(optimizer, feed_dict = feed_dict)

		if step % disp_step == 0:
			# Calculate batch accuracy
			acc = sess.run(accuracy, feed_dict = feed_dict)
			# Calculate batch loss
			cost = sess.run(loss, feed_dict = feed_dict)
			print(
				"Iter " + str(step * batch_size) + ", Loss = " + \
				"%6f" % cost + ", Accuracy = " + \
				"%6f" % acc \
			)
		# """
		org = np.array(sess.run(y_one_col, feed_dict = feed_dict)).reshape([-1, seq_length - 1])
		pred = np.array(sess.run(tf.argmax(pred_logits, 1), feed_dict = feed_dict)).reshape([-1, seq_length - 1])

		
		for i in range(org.shape[0]):
			a = ""
			b = ""
			for j in range(org.shape[1]):
				a += (look_up[org[i, j]] + " ")
				b += (look_up[pred[i, j]] + " ")
			print(a)
			print(b)
		# """
		step += 1

		# state_feed = sess.run(state, feed_dict = feed_dict)

	print("Optimization Finished!")

	save_path = saver.save(sess, "../lizuoyue-loop/model.ckpt")
	print("Model saved in file: %s" % save_path)
