# ETH Zurich, Semester S17
# Natural Language Understanding, Task 1(B)
# Team Members: Jie Huang, Yanping Xie, Zuoyue Li

from __future__ import print_function

# Deactivate the warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Import packages
import tensorflow as tf
import numpy as np
from gensim import models

print("Import packages ... Done!")

# Set learning parameters
learning_rate  = 0.001 # learning rate
training_iters = 2e7   # training iters
clip_norm      = 10.0  # global norm
disp_step      = 10    # display step

# Set network parameters
batch_size     = 64    # batch size
vocab_size     = 20000 # vocabulary size
emb_size       = 100   # word embedding size
seq_length     = 30    # sequence length
state_size     = 512   # hidden state size
model_save     = 600   # save per number of batches
emb_path       = "../data/wordembeddings-dim100.word2vec"

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

def load_embedding(session, vocab, emb, path, dim_embedding):
	'''
	  session        Tensorflow session object
	  vocab          A dictionary mapping token strings to vocabulary IDs
	  emb            Embedding tensor of shape (vocabulary_size, dim_embedding)
	  path           Path to embedding file
	  dim_embedding  Dimensionality of the external embedding.
	'''
	print("Loading external embeddings from %s" % path)
	model = models.KeyedVectors.load_word2vec_format(path, binary = False)

	external_embedding = np.zeros(shape = (vocab_size, dim_embedding))
	matches = 0
	for tok, idx in vocab.items():
		if tok in model.vocab:
			external_embedding[idx] = model[tok]
			matches += 1
		else:
			print("%s not in embedding file" % tok)
			external_embedding[idx] = np.random.uniform(low = -0.25, high = 0.25, size = dim_embedding)

	print("%d words out of %d could be loaded" % (matches, vocab_size))

	pretrained_embeddings = tf.placeholder(tf.float32, [None, None]) 
	assign_op = emb.assign(pretrained_embeddings)
	session.run(assign_op, {pretrained_embeddings: external_embedding})

# Define RNN network input and output
x = tf.placeholder(tf.int32, [batch_size, seq_length       ])
y = tf.placeholder(tf.int32, [batch_size * (seq_length - 1)])

# Define word embeddings, output weight and output bias
emb_weight  = tf.get_variable("emb_weight", [vocab_size, emb_size  ], dtype = tf.float32, trainable = True)
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
lstm_cell   = tf.contrib.rnn.BasicLSTMCell(state_size)
init_state  = lstm_cell.zero_state(batch_size, tf.float32)
state       = init_state
output_seq  = []
time_step =0
with tf.variable_scope("RNN"):
	for input_unit in input_seq:
		if time_step > 0: tf.get_variable_scope().reuse_variables()
		time_step+=1

		output_unit, state = lstm_cell(input_unit, state)
		output_seq.append(output_unit)
output_seq.pop()
final_state = state
output_seq  = tf.reshape(output_seq, [-1, state_size])
pred_logits = tf.matmul(output_seq, out_weight) + out_bias

print("Define network computation process ... Done!")

# Define loss and optimizer
loss        = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = pred_logits, labels = y))
opt_func    = tf.train.AdamOptimizer(learning_rate = learning_rate)
grad, var   = zip(*opt_func.compute_gradients(loss))
grads, _    = tf.clip_by_global_norm(grad, clip_norm)
optimizer   = opt_func.apply_gradients(zip(grads, var))

# Initialize the variables
init        = tf.global_variables_initializer()
saver       = tf.train.Saver()

print("Define loss, optimizer and evaluate function ... Done!")

# Launch the graph
print("Start training!")
out = open("lb-log.txt","w")
f = open("../data/sentences.train", 'r')
NUM_THREADS=8
with tf.Session(config=tf.ConfigProto(inter_op_parallelism_threads=NUM_THREADS,intra_op_parallelism_threads=NUM_THREADS)) as sess:

	sess.run(init)
	load_embedding(sess, vocabulary, emb_weight, emb_path, emb_size)
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
				code.append(vocabulary["<eos>"])
				while len(code) < seq_length:
					code.append(vocabulary["<pad>"])
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

		if step >= 1:
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
			out.write(str(step * batch_size)+" "+str(cost)+" "+str(perp)+"\n")
			out.flush()

			# Print prediction
			"""
			pred = np.array(sess.run(tf.argmax(pred_logits, 1), feed_dict = feed_dict)).reshape([-1, batch_size]).transpose()
			for i in range(pred.shape[0]):
				a = ""
				b = ""
				for j in range(pred.shape[1]):
					a += (look_up[batch_x[i, j + 1]] + " ")
					b += (look_up[pred[i, j]] + " ")
				print("# " + a + "\n")
				print("@ " + b + "\n")
				break
			"""

		if step % model_save == 0:
			save_path = saver.save(sess, "../li-b-" + str(step) + ".ckpt")

		# state_feed = sess.run(final_state, feed_dict = feed_dict)
		step += 1

	print("Optimization Finished!")
	model_path = "../final-lib.ckpt"
	save_path = saver.save(sess, model_path)
	print("Model saved in file: %s" % save_path)
out.close()
f.close()
