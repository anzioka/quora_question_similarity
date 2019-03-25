from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Dropout, LSTM, Add, Concatenate, Input, BatchNormalization, TimeDistributed
from keras.layers.embeddings import Embedding
from keras.layers.wrappers import Bidirectional
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras import regularizers

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import *

def model(config, embeddings):
	print("Building model")

	question1 = Input(shape=(config['seq_len'],))
	question2 = Input(shape=(config['seq_len'],))

	q1 = Embedding(config['vocab_size'], config['embed_dim'], input_length = config['seq_len'], weights=[embeddings], mask_zero=True, trainable=False) (question1)
	q1 = Bidirectional(LSTM(config['embed_dim'], return_sequences=True))(q1) #hidden size can be different but we are choosing same size as the size of the embedding vector
	q1 = Bidirectional(LSTM(config['embed_dim'], return_sequences=True))(q1)
	q1 = Bidirectional(LSTM(config['embed_dim']))(q1)

	q2 = Embedding(config['vocab_size'], config['embed_dim'], input_length = config['seq_len'], weights=[embeddings], mask_zero=True, trainable=False) (question2)
	q2 = Bidirectional(LSTM(config['embed_dim'], return_sequences=True))(q2) #output dimension is 100
	q2 = Bidirectional(LSTM(config['embed_dim'], return_sequences=True))(q2)
	q2 = Bidrectional(LSTM(config['embed_dim']))(q2)

	#first two LSTMs stacked to learn higher-level temporal representation
	#batch normalization to quicken training
	merged = Concatenate()([q1, q2])
	merged = Dense(200, activation = "relu", kernel_regularizer=regularizers.l2(config['l2']))(merged)
	merged = Dropout(config['dropout']) (merged)
	merged = BatchNormalization() (merged)
	merged = Dense(200, activation = "relu", kernel_regularizer=regularizers.l2(config['l2']))(merged)
	merged = Dropout(config['dropout']) (merged)
	merged = BatchNormalization() (merged)
	merged = Dense(2, activation = "softmax") (merged)
	model = Model(inputs=[question1, question2], outputs=merged)

	optimizer = Adam(lr=config['lr'])
	model.compile(optimizer=optimizer, loss=config['loss'], metrics=["accuracy", precision, recall, f1_score])
	model.summary()
	return model
	

