import tensorflow as tf
import numpy as np
import os
import random
import sys
sys.path.append('../')
from aa.gensim import models
#from load_embeddings import load_embedding
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


word_embedding_size = 100
hidden_size = 512
num_steps = 30
keep_prob = 1
batch_size = 64
vocab_size = 20000
training_iters = 100000
display_step = 1
learning_rate = 0.01
num_layers = 2

def load_embedding(session, vocab, emb, path, dim_embedding):
    '''
      session        Tensorflow session object
      vocab          A dictionary mapping token strings to vocabulary IDs
      emb            Embedding tensor of shape vocabulary_size x dim_embedding
      path           Path to embedding file
      dim_embedding  Dimensionality of the external embedding.
    '''
    print("Loading external embeddings from %s" % path)
    model = models.KeyedVectors.load_word2vec_format(path, binary=False)
    external_embedding = np.zeros(shape=(vocab_size, dim_embedding))
    matches = 0
    for tok, idx in vocab.items():
        if tok in model.vocab:
            external_embedding[idx] = model[tok]
            matches += 1
        else:
            print("%s not in embedding file" % tok)
            external_embedding[idx] = np.random.uniform(low=-0.25, high=0.25, size=dim_embedding)
    print (len(external_embedding))
    print("%d words out of %d could be loaded" % (matches, vocab_size))
    return external_embedding
    #pretrained_embeddings = tf.placeholder(tf.float32, [vocab_size, word_embedding_size]) 
    #assign_op = tf.assign(emb,pretrained_embeddings)
    #session.run(assign_op, {pretrained_embeddings: external_embedding})





lstm_cell = tf.contrib.rnn.BasicLSTMCell(hidden_size, forget_bias=0.0, state_is_tuple=True,reuse=True)
#lstm_cell = tf.contrib.rnn.BasicRNNCell(hidden_size)
lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=keep_prob)

#lstm_cell2 = tf.contrib.rnn.BasicLSTMCell(hidden_size, forget_bias=0.0, state_is_tuple=True)
#lstm_cell = tf.contrib.rnn.BasicRNNCell(hidden_size)
#lstm_cell2 = tf.contrib.rnn.DropoutWrapper(lstm_cell2, output_keep_prob=keep_prob)
#cell = tf.contrib.rnn.MultiRNNCell([lstm_cell,lstm_cell2], state_is_tuple=True)
cell = lstm_cell

#initial_state = cell.zero_state(batch_size, tf.float32)
embedding = tf.placeholder(tf.float32, [vocab_size, word_embedding_size])
#tf.get_variable("embedding", [vocab_size, word_embedding_size], initializer = tf.contrib.layers.xavier_initializer())
input_data = tf.placeholder(tf.int32, [batch_size, num_steps])
targets = tf.placeholder(tf.int32, [batch_size, num_steps-1])
inputs = tf.nn.embedding_lookup(embedding, input_data)   #batch_size * num_steps * word_embedding_size

#w1 = tf.get_variable(
#        "w1", [word_embedding_size, hidden_size], dtype=tf.float32)
#b1 = tf.get_variable("b1", [hidden_size], dtype=tf.float32)

with tf.variable_scope("basic_lstm_cell"):
	weights = tf.get_variable("weights", [ word_embedding_size + hidden_size, 4 * hidden_size], dtype = tf.float32, initializer = tf.contrib.layers.xavier_initializer())
	biases  = tf.get_variable("biases" , [4 * hidden_size], dtype = tf.float32, initializer = tf.contrib.layers.xavier_initializer())


outputs = []
state = cell.zero_state(batch_size, tf.float32)
#with tf.variable_scope("RNN"):
for time_step in range(num_steps-1):
	#if time_step > 0: tf.get_variable_scope().reuse_variables()
	#act_input = tf.matmul(inputs[:, time_step, :], w1) + b1
	#(cell_output, state) = cell(act_input, state)
	(cell_output, state) = cell(inputs[:, time_step, :], state)
	outputs.append(cell_output)

output = tf.reshape(tf.concat(outputs,1), [-1, hidden_size])
softmax_w = tf.get_variable(
        "softmax_w", [hidden_size, vocab_size], dtype=tf.float32, initializer = tf.contrib.layers.xavier_initializer())
softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=tf.float32, initializer = tf.contrib.layers.xavier_initializer())
logits = tf.matmul(output, softmax_w) + softmax_b

loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.reshape(targets, [-1]), logits=logits))
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
gradients, variables = zip(*optimizer.compute_gradients(loss))
gradients, _ = tf.clip_by_global_norm(gradients, 10.0)
train_op = optimizer.apply_gradients(zip(gradients, variables))


pred = tf.reshape(logits,[batch_size , (num_steps-1), vocab_size])
correct_pred = tf.equal(tf.argmax(pred, 2),tf.to_int64(targets)) 
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))



# Initialize the variables
init = tf.global_variables_initializer()

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
NUM_THREADS = 4

#emb = tf.Variable(initial_value=tf.zeros([vocab_size, word_embedding_size]), name='emb',dtype=tf.float32)
#emb = tf.Session().run(emb)

emb = load_embedding(session=tf.Session(config=tf.ConfigProto(inter_op_parallelism_threads=NUM_THREADS,intra_op_parallelism_threads=NUM_THREADS)),
	vocab=vocabulary,
	emb = "",
	path = "../data/wordembeddings-dim100.word2vec",
	dim_embedding = word_embedding_size)

print ("load embedding success")
#print (emb.shape)
#for i in range(200):
#	for j in range(100):
#		print (emb[i][j])
#test = np.zeros([200,100])

#f = open("../data/sentences.train", 'r')
input_file = "../data/sentences.train"
#input_file = "../data/1.txt"

f = open(input_file, 'r')
out = open("log.txt",'w')
# Launch the graph
print("Start Training!")

with tf.Session(config=tf.ConfigProto(inter_op_parallelism_threads=NUM_THREADS,intra_op_parallelism_threads=NUM_THREADS)) as sess:

	sess.run(init)
	step = 1
	# Keep training until reach max iterations
	while step * batch_size < training_iters:

		batch_x = []
		while len(batch_x) < batch_size:
			line = f.readline()
			if not line or step % 100==0:
				f.close()

				f = open(input_file, 'r')
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


		# Random generation of input data
		'''
		batch_x = []
		for k in range(batch_size):
			code = [random.randint(0, vocab_size - 1)]
			for i in range(num_steps - 1):
				code.append((code[i] + 1) % vocab_size)
			batch_x.append(code)
		batch_x = np.array(batch_x)
		'''
		batch_y = np.zeros((batch_size, num_steps - 1))
		for i in range(batch_size):
			for j in range(1, num_steps):
				batch_y[i, j - 1] =  batch_x[i, j]
	
		feed_dict = {input_data: batch_x, targets: batch_y, embedding : emb}
		sess.run(train_op, feed_dict = feed_dict)



		# print("Optimize Done!")
		
		if step % display_step == 0:
			print(sess.run(targets,feed_dict = feed_dict))
			tmp = sess.run(tf.argmax(pred, 2), feed_dict = feed_dict)
			print(tmp[0:10][0:10])
			out.write(str(tmp[0:10][0:10])+"\n")
			mloss = sess.run(loss, feed_dict = feed_dict)
			#acc = sess.run(,feed_dict = {input_data: batch_x, targets: batch_y})
			[logit,acc] = sess.run([logits,accuracy],feed_dict =feed_dict)

			print (logit[0:10][0:10])
			out.write(str(logit[0:10][0:10])+"\n")
			print("Iter " + str(step * batch_size) + ", Minibatch Loss= " + str(mloss))
			out.write(str(acc)+"\n")
			out.flush();
			print ("acc: " + str(acc))
		
		step += 1
	print("Optimization Finished!")
	saver = tf.train.Saver( )
	fname = str(word_embedding_size)+'-'+str(hidden_size)+'-'+str(batch_size)+'-'+str(vocab_size)+'-'+str(learning_rate)+'-'+str(training_iters)
	save_path = saver.save(sess, './model/lstm'+fname+'.ckpt')

	# Calculate accuracy for test set
	
out.close()







'''

for i in range(max_epoch):
  _, final_state = session.run([train_op, state],
                               {input_data: x,
                                targets: y})
'''