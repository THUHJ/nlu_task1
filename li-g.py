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

# Set network parameters
batch_size  = 1
vocab_size  = 20000 # vocabulary size
emb_size    = 100   # word embedding size
seq_length  = 20    # sequence length
state_size  = 512   # hidden state size
model_path  = "../li-b-7200.ckpt"

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
	line = f.readline()
f.close()

print("Load dictionary ... Done!")

# Define RNN network input and output
x = tf.placeholder(tf.int32, [batch_size])

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
lstm_cell   = tf.contrib.rnn.BasicLSTMCell(state_size, reuse = True)
init_state  = lstm_cell.zero_state(batch_size, tf.float32)
state       = init_state
out, state  = lstm_cell(input_emb, state)
final_state = state
pred_logits = tf.matmul(out, out_weight) + out_bias
next_word   = tf.argmax(pred_logits, 1)

# Initialize the variables
saver       = tf.train.Saver()

print("Define network computation process ... Done!")

# Launch the graph
print("Start generation!")

with tf.Session() as sess:

	saver.restore(sess, model_path)

	f = open("../data/sentences.continuation", 'r')
	line = f.readline()

	while line:

		step = 1
		words = line.strip().split(' ')
		code = [vocabulary["<bos>"]]
		for word in words:
			if word in vocabulary:
				code.append(vocabulary[word])
			else:
				code.append(vocabulary["<unk>"])

		for idx in code:
			if step == 1:
				feed_dict = {x: np.array([idx])}
			else:
				feed_dict = {x: np.array([idx]), init_state: state_feed}

			next_idx = sess.run(next_word, feed_dict = feed_dict)
			state_feed = sess.run(final_state, feed_dict = feed_dict)
			step += 1

		next_words = ""
		for i in range(len(code), seq_length):
			feed_dict = {x: next_idx, init_state: state_feed}
			next_idx = sess.run(next_word, feed_dict = feed_dict)
			state_feed = sess.run(final_state, feed_dict = feed_dict)
			if next_idx[0] != vocabulary["<eos>"]:
				next_words += look_up[next_idx[0]]
				next_words += " "
			else:
				break
		print(line.strip() + " @@@ " + next_words)
		line = f.readline()

	f.close()

	print("Prediction finished!")
