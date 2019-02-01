from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Dropout, LSTM, Add, concatenate, Input, BatchNormalization
from keras.layers.embeddings import Embedding
from keras.layers.wrappers import Bidirectional
from keras.models import Sequential, Model



def model(config, embeddings):
	print("Building model")

	question1 = Input(shape=(config.seq_len,))
	question2 = Input(shape=(config.seq_len,))

	q1 = Embedding(config.vocab_size, config.embed_dim, input_length = config.seq_len) (question1)
	q1 = LSTM(config.embed_dim)(q1) #hidden size can be different but we are choosing same size as the size of the embedding vector
	q1 = Dropout(config.dropout)(q1)

	q2 = Embedding(config.vocab_size, config.embed_dim, input_length = config.seq_len) (question2)
	q2 = LSTM(config.embed_dim)(q2) #output dimension is 100
	q2 = Dropout(config.dropout)(q2)

	merged = concatenate([q1, q2])
	merged = Dense(200, activation = "relu")(merged)
	merged = BatchNormalization() (merged)
	merged = Dense(200, activation = "relu")(merged)
	merged = BatchNormalization() (merged)

	merged = Dense(1, activation = "softmax") (merged)

	model = Model(inputs=[question1, question2], outputs=merged)
	model.compile(optimizer=config.optimizer, loss=config.loss, metrics=["accuracy"])
	model.summary()
	return model
	









