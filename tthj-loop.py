import tensorflow as tf
import numpy as np


size = 256
num_steps = 30
keep_prob = 1
batch_size = 64
vocab_size = 200
training_iters = 10000
display_step = 1
learning_rate = 3e-3



lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(size, forget_bias=0.0,state_is_tuple=True)
lstm_cell = tf.nn.rnn_cell.DropoutWrapper(
    lstm_cell, output_keep_prob=keep_prob)
#cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * num_steps, state_is_tuple=True)
cell = lstm_cell

initial_state = cell.zero_state(batch_size, tf.float32)
embedding = tf.get_variable("embedding", [vocab_size, size])
# input_data: [batch_size, num_steps]
# targets： [batch_size, num_steps]
input_data = tf.placeholder(tf.int32, [batch_size, num_steps])
targets = tf.placeholder(tf.int32, [batch_size, num_steps-1])
inputs = tf.nn.embedding_lookup(embedding, input_data)
outputs = []
state = initial_state
with tf.variable_scope("RNN"):
	for time_step in range(num_steps-1):
		if time_step > 0: tf.get_variable_scope().reuse_variables()
		(cell_output, state) = cell(inputs[:, time_step, :], state)
		outputs.append(cell_output)

output = tf.reshape(tf.concat(1, outputs), [-1, size])
softmax_w = tf.get_variable(
        "softmax_w", [size, vocab_size], dtype=tf.float32)
softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=tf.float32)
logits = tf.matmul(output, softmax_w) + softmax_b

#loss = tf.nn.seq2seq.sequence_loss_by_example(
#    [logits],
#    [tf.reshape(targets, [-1])],
#    [tf.ones([batch_size * (num_steps-1)])])

loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.reshape(targets, [-1]), logits=logits)

optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
gradients, variables = zip(*optimizer.compute_gradients(loss))
gradients, _ = tf.clip_by_global_norm(gradients, 10.0)
train_op = optimizer.apply_gradients(zip(gradients, variables))

pred = tf.reshape(logits,[batch_size , (num_steps-1), vocab_size])
correct_pred = tf.equal(tf.argmax(pred, 2),tf.to_int64(targets)) 
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


# Initialize the variables
init = tf.initialize_all_variables()

# Construct vocabulary index dictionary
vocabulary = {}
f = open("vocabulary.txt", 'r')
line = f.readline()
idx = 0
while line:
	vocabulary[line.strip()] = idx;
	idx += 1
	if vocab_size==idx:
		break
	line  = f.readline()
f.close()

#f = open("../data/sentences.train", 'r')
f = open("../data/test.train", 'r')

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
			if (len(words) <= num_steps-2):
				code = [vocabulary["<bos>"]]
				for word in words:
					if word in vocabulary:
						code.append(vocabulary[word])
					else:
						code.append(vocabulary["<unk>"])
				while (len(code) < num_steps-1):
					code.append(vocabulary["<pad>"])
				code.append(vocabulary["<eos>"])
				batch_x.append(code)

		batch_x = np.array(batch_x)
		# batch_x = np.zeros((batch_size, n_steps), dtype = np.int32)
		batch_y = np.zeros((batch_size, num_steps - 1))
		for i in range(batch_size):
			# batch_x[i, 0] = random.randint(0, n_vocab - 1)
			for j in range(1, num_steps):
				# batch_x[i, j] = (batch_x[i, j - 1] + 1) % n_vocab
				batch_y[i, j - 1] =  batch_x[i, j]
		# print(batch_x)
		# print(batch_y)
		# print("Data Done!")

		sess.run(train_op, feed_dict = {input_data: batch_x, targets: batch_y})
		# print("Optimize Done!")
		
		if step % display_step == 0:
			# Calculate batch accuracy
			#acc = sess.run(accuracy, feed_dict = {input_data: batch_x, targets: batch_y})
			# Calculate batch loss
			#mloss = sess.run(loss, feed_dict = {input_data: batch_x, targets: batch_y})
			acc = sess.run(accuracy,feed_dict = {input_data: batch_x, targets: batch_y})
			print(
				"Iter " + str(step * batch_size) + ", Minibatch Loss= " 
			)
			#print (mloss)
			print ("acc: " + str(acc))
		
		step += 1
	print("Optimization Finished!")

	# Calculate accuracy for test set
	








'''

for i in range(max_epoch):
  _, final_state = session.run([train_op, state],
                               {input_data: x,
                                targets: y})
'''
