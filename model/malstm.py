from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Dropout, LSTM, Add, Input, Lambda
from keras.layers.embeddings import Embedding
from keras.layers.wrappers import Bidirectional
from keras.models import Sequential, Model
from keras.optimizers import Adam, Adadelta
from keras import regularizers

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import *

def model(config, embeddings):

	gradient_clipping_norm = 1.25
	question1 = Input(shape=(config['seq_len'],))
	question2 = Input(shape=(config['seq_len'],))

	embedding_layer = Embedding(config['vocab_size'], config['embed_dim'], weights=[embeddings], input_length=config['seq_len'], trainable=False)
	q1 = embedding_layer(question1)
	q2 = embedding_layer(question2)

	shared_lstm = LSTM(config['hidden_dim'])
	q1 = shared_lstm(q1)
	q2 = shared_lstm(q2)

	#calculate the distance 
	distance = Lambda(function=lambda x: exponent_neg_manhattan_distance(x[0], x[1]),output_shape=lambda x: (x[0][0], 1))([q1, q2])
	model = Model([question1, question2], [distance])

	optimizer = Adadelta(clipnorm=gradient_clipping_norm)
	model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=["accuracy", precision, recall, f1_score])
	model.summary()
	return model


