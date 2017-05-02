import tensorflow as tf
import numpy as np


word_embedding_size = 100
hidden_size = 512
num_steps = 30
keep_prob = 1
batch_size = 1
vocab_size = 2000
training_iters = 1000
display_step = 1
learning_rate = 0.01



lstm_cell = tf.contrib.rnn.BasicLSTMCell(hidden_size, forget_bias=0.0,state_is_tuple=True, reuse=True)
lstm_cell = tf.contrib.rnn.DropoutWrapper(
    lstm_cell, output_keep_prob=keep_prob)
#cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * num_steps, state_is_tuple=True)
cell = lstm_cell

state = cell.zero_state(batch_size, tf.float32)

embedding = tf.get_variable("embedding", [vocab_size, word_embedding_size], initializer = tf.contrib.layers.xavier_initializer())
input_one = tf.placeholder(tf.int32, [batch_size])
input_one_embedding = tf.nn.embedding_lookup(embedding, input_one) #batch_size  * word_embedding_size

with tf.variable_scope("basic_lstm_cell"):
	weights = tf.get_variable("weights", [word_embedding_size + hidden_size, 4 * hidden_size], dtype = tf.float32, initializer = tf.contrib.layers.xavier_initializer())
	biases  = tf.get_variable("biases" , [4 * hidden_size], dtype = tf.float32, initializer = tf.contrib.layers.xavier_initializer())

softmax_w = tf.get_variable(
        "softmax_w", [hidden_size, vocab_size], dtype=tf.float32, initializer = tf.contrib.layers.xavier_initializer())
softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=tf.float32, initializer = tf.contrib.layers.xavier_initializer())



(cell_output, state) = cell(input_one_embedding, state)
logits = tf.matmul(cell_output, softmax_w) + softmax_b
pred = tf.reshape(logits,[batch_size , vocab_size])
next_word = tf.argmax(pred, 1)


'''
outputs = []
state = initial_state
with tf.variable_scope("RNN"):
	for time_step in range(num_steps-1):
		if time_step > 0: tf.get_variable_scope().reuse_variables()
		(cell_output, state) = cell(inputs[:, time_step, :], state)
		outputs.append(cell_output)

output = tf.reshape(tf.concat(outputs,1), [-1, hidden_size])

logits = tf.matmul(output, softmax_w) + softmax_b

#loss = tf.nn.seq2seq.sequence_loss_by_example(
#    [logits],
#    [tf.reshape(targets, [-1])],
#    [tf.ones([batch_size * (num_steps-1)])])

loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.reshape(targets, [-1]), logits=logits))

optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
gradients, variables = zip(*optimizer.compute_gradients(loss))
gradients, _ = tf.clip_by_global_norm(gradients, 10.0)
train_op = optimizer.apply_gradients(zip(gradients, variables))

pred = tf.reshape(logits,[batch_size , (num_steps-1), vocab_size])
correct_pred = tf.equal(tf.argmax(pred, 2),tf.to_int64(targets)) 
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
'''


# Construct vocabulary index dictionary
vocabulary = {}
revovocabulary = {}
f = open("vocabulary.txt", 'r')
line = f.readline()
idx = 0
while line:
	vocabulary[line.strip()] = idx;
	revovocabulary[idx] = line.strip()
	idx += 1
	if vocab_size==idx:
		break
	line  = f.readline()
f.close()

# Launch the graph
print("Start Training!")
NUM_THREADS = 4
with tf.Session(config=tf.ConfigProto(inter_op_parallelism_threads=NUM_THREADS,intra_op_parallelism_threads=NUM_THREADS)) as sess:

	saver = tf.train.Saver( )
	fname = str(word_embedding_size)+'-'+str(hidden_size)+'-'+str(64)+'-'+str(vocab_size)+'-'+str(learning_rate)+'-'+str(training_iters)
	saver.restore(sess, './model/lstm'+fname+'.ckpt')

	step = 1
	# Keep training until reach max iterations
	f = open("../data/sentences.continuation", 'r')
	while line:
		words = line.strip().split(' ')
		
		code = [vocabulary["<bos>"]]
		for word in words:
			if word in vocabulary:
				code.append(vocabulary[word])
			else:
				code.append(vocabulary["<unk>"])
		#while (len(code) < num_steps-1):
		#	code.append(vocabulary["<pad>"])
		#code.append(vocabulary["<eos>"])

		my_next_word = ""
		last_state = ""
		for i in code:
			feed_dict={input_one:np.array([i])}

			if (last_state!=""):
				feed_dict={input_one:np.array([i]),state:last_state}
				
			my_next_word = sess.run(next_word,feed_dict=feed_dict)
			last_state = sess.run(state,feed_dict=feed_dict)
			print (revovocabulary[i],end=" ")
		
		for i in range(len(code),20):
			feed_dict={input_one: my_next_word,state:last_state}
			my_next_word = sess.run(next_word,feed_dict=feed_dict)
			last_state = sess.run(state,feed_dict=feed_dict)
			print (revovocabulary[my_next_word[0]],end=" ")
			if (my_next_word==1):
				break
		print ("");
		line = f.readline()
	f.close()

