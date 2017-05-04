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

print("Import packages ... Done!")

# Set learning parameters
learning_rate  = 5e-2  # learning rate
training_iters = 2e4   # training iters
global_norm    = 10.0  # global norm
disp_step      = 5     # display step

# Set network parameters
batch_size     = 64    # batch size
vocab_size     = 20000 # vocabulary size
emb_size       = 100   # word embedding size
seq_length     = 30    # sequence length
state_size     = 512   # hidden state size
model_path     = "../li-a-model.ckpt"

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

# Define RNN network input and output
x = tf.placeholder(tf.int32, [batch_size, seq_length       ])
y = tf.placeholder(tf.int32, [batch_size * (seq_length - 1)])

# Define word embeddings, output weight and output bias
emb_weight  = tf.get_variable("emb_weight", [vocab_size, emb_size  ], dtype = tf.float32, initializer = tf.contrib.layers.xavier_initializer())
out_weight  = tf.get_variable("out_weight", [state_size, vocab_size], dtype = tf.float32, initializer = tf.contrib.layers.xavier_initializer())
out_bias    = tf.get_variable("out_bias"  , [vocab_size]            , dtype = tf.float32, initializer = tf.contrib.layers.xavier_initializer())

# Define LSTM cell weights and biases
with tf.variable_scope("basic_lstm_cell"):
	weights = tf.get_variable("weights", [emb_size + state_size, 4 * state_size], \
				dtype = tf.float32, initializer = tf.contrib.layers.xavier_initializer())
	biases  = tf.get_variable("biases" , [4 * state_size], dtype = tf.float32, initializer = tf.contrib.layers.xavier_initializer())

print("Define network parameters ... Done!")

# Define RNN computation process
input_emb   = tf.nn.embedding_lookup(emb_weight, x)
input_seq   = tf.unstack(input_emb, axis = 1)
lstm_cell   = tf.contrib.rnn.BasicLSTMCell(state_size, forget_bias = 0.0)
init_state  = lstm_cell.zero_state(batch_size, tf.float32)

output_seq  = []
time_step=0
with tf.variable_scope("rnn"):
	state       = init_state
	for input_unit in input_seq:
		if time_step > 0: tf.get_variable_scope().reuse_variables()
		time_step=1
		output_unit, state = lstm_cell(input_unit, state)
		output_seq.append(output_unit)
	output_seq.pop()
final_state = state
# 29 * 64 * 512
#print (tf.concat(output_seq,1).shape)

output_seq  = tf.reshape(tf.concat(output_seq,1), [-1, state_size])

pred_logits = tf.matmul(output_seq, out_weight) + out_bias
print (pred_logits.shape)
print (y.shape)

print("Define network computation process ... Done!")

# Define loss and optimizer
loss        = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = pred_logits, labels = y))
opt_func    = tf.train.AdamOptimizer(learning_rate = learning_rate)
grad, var   = zip(*opt_func.compute_gradients(loss))
grad, _     = tf.clip_by_global_norm(grad, global_norm)
optimizer   = opt_func.apply_gradients(zip(grad, var))

# Initialize the variables
init        = tf.global_variables_initializer()
saver       = tf.train.Saver()

print("Define loss, optimizer and evaluate function ... Done!")

# Launch the graph
print("Start training!")

f = open("../data/sentences.train", 'r')
with tf.Session() as sess:

	sess.run(init)
	step = 1

	while step * batch_size < training_iters:

		batch_x = []
		while len(batch_x) < batch_size:

			line = f.readline()
			if not line:
				f.close()
				f = open("../data/sentences.train", 'r')
				line = f.readline()

			words = line.strip().split(' ')
			if len(words) < seq_length - 1:
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
		import random
		batch_x = []
		for k in range(batch_size):
			code = [random.randint(0, vocab_size - 1)]
			for i in range(seq_length - 1):
				code.append((code[i] + 1) % vocab_size)
			batch_x.append(code)
		"""

		batch_x = np.array(batch_x)
		batch_m = batch_x[:, 1: seq_length].transpose()
		batch_y = batch_m.reshape([-1])

		if step == 1:
			feed_dict = {x: batch_x, y: batch_y}
		else:
			feed_dict = {x: batch_x, y: batch_y, init_state: state_feed}

		sess.run(optimizer, feed_dict = feed_dict)

		# Evaluate model
		if step % disp_step == 0:
			cost = sess.run(loss, feed_dict = feed_dict)
			prob = sess.run(tf.nn.softmax(pred_logits), feed_dict = feed_dict)
			prob = np.array(prob).reshape([seq_length - 1, batch_size, vocab_size])
			psum = [0.0 for i in range(batch_size)]
			pnum = [0.0 for i in range(batch_size)]
			for i in range(seq_length - 1):
				for j in range(batch_size):
					if batch_m[i, j] != vocabulary["<pad>"]:
						psum[j] += np.log(prob[i, j, batch_m[i, j]])
						pnum[j] += 1.0
			perp_list = [2 ** (-psum[i] / pnum[i]) for i in range(batch_size)]
			perp = sum(perp_list) / len(perp_list)
			print(
				"Iter " + str(step * batch_size) + \
				", Loss = %6f" % cost + \
				", Perp = %6f" % perp \
			)

			# Print prediction
			# """
			pred = np.array(sess.run(tf.argmax(pred_logits, 1), feed_dict = feed_dict)).reshape([-1, batch_size]).transpose()
			for i in range(pred.shape[0]):
				a = ""
				b = ""
				for j in range(pred.shape[1]):
					a += (look_up[batch_x[i, j + 1]] + " ")
					b += (look_up[pred[i, j]] + " ")
				print("# " + a + "\n")
				print("@ " + b + "\n")
			# """

		state_feed = sess.run(final_state, feed_dict = feed_dict)
		step += 1

	print("Optimization Finished!")

	save_path = saver.save(sess, model_path)
	print("Model saved in file: %s" % save_path)

f.close()
