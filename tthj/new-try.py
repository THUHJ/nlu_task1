# Import packages
import tensorflow as tf
import numpy as np

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

f=open("vocabulary.txt",'r')
line = f.readline()
while line:
	print (line.strip())
	line = f.readline()
