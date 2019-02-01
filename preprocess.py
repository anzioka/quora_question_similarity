
import os
import random
import requests
import numpy as np
import json
import csv
import pandas as pd
from tqdm import tqdm
from keras.utils.data_utils import get_file
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from zipfile import ZipFile
from io import BytesIO
from urllib.request import urlopen

glove_url = 'http://nlp.stanford.edu/data/glove.840B.300d.zip'
quora_url = 'http://qim.fs.quoracdn.net/quora_duplicate_questions.tsv'
glove_embeddings = 'data/glove/glove.840B.300d.txt'
quora_dataset = 'data/quora/quora_duplicate_questions.tsv'
word_embeddings_path = "data/embeddings/embeddings.npy"
q1_train_path= "data/train/q1_train.npy"
q2_train_path = "data/train/q2_train.npy"
q1_dev_path = "data/dev/q1_dev.npy"
q2_dev_path = "data/dev/q2_dev.npy"
q1_test_path = "data/test/q1_test.npy"
q2_test_path = "data/test/q2_test.npy"
labels_train_path = "data/labels/labels_train.npy"
labels_dev_path = "data/labels/labels_dev.npy"
labels_test_path = "data/labels/labels_test.npy"
dataset_params = "data/dataset_params.json"
max_seq = 25
embed_dim = 300

def save_json(dataset_info):
	'''saves file to json 
	'''
	with open(dataset_params, 'w') as f:
		d = {k : v for k, v in dataset_info.items()}
		json.dump(d, f, indent=4)

def process_dataset():
	q1_data = []
	q2_data = []
	labels = []

	print("Processing dataset...")
	with open(quora_dataset, encoding = 'utf-8') as f:
		reader = csv.DictReader(f, delimiter='\t');
		for row in tqdm(reader):
			q1_data.append(row['question1'])
			q2_data.append(row['question2'])
			labels.append(row['is_duplicate'])

	assert len(labels) == len(q2_data) == len(q1_data)

	#pad sequences
	t = Tokenizer()
	t.fit_on_texts(q1_data + q2_data)
	q_1 = t.texts_to_sequences(q1_data)
	q_2 = t.texts_to_sequences(q2_data)

	print (q1_data[:10])
	print (q_1[:10])

	q_1 = pad_sequences(q_1, maxlen = max_seq, padding="post")
	q_2 = pad_sequences(q_2, maxlen = max_seq, padding="post")
	assert q_2.shape == q_1.shape

	print ("padded q1 shape : {}".format(q_1.shape))

	#create embedding dictionary
	word_index = t.word_index
	vocab_size = len(word_index) + 1

	print ("vocab_size: {}".format(vocab_size))
	print ("Creating embedding dictionary...")
	embedding_dict = {}
	temp = []
	with open(glove_embeddings, encoding='utf-8') as f:
		for line in tqdm(f):
			values = line.split(' ')
			embedding_dict[values[0]] = np.asarray(values[1:], dtype="float32")

	print ("Creating word embeddings...")
	word_embeddings = np.zeros((vocab_size, embed_dim))
	for word, i in tqdm(word_index.items()):
		e = embedding_dict.get(word)
		if e is not None:
			word_embeddings[i] = e
	
	print ("Shape of word word_embeddings: " + str(word_embeddings.shape))

	#split into 80% training, 10%dev and 10%test
	split_1 = int(0.8 * len(q_1))
	split_2 = int(0.9 * len(q_1))

	q1_train = q_1[:split_1]
	q2_train = q_2[:split_1]

	q1_dev = q_1[split_1 : split_2]
	q2_dev = q_2[split_1 : split_2]

	q1_test = q_1[split_2:]
	q2_test = q_2[split_2:]

	labels_train = labels[:split_1]
	labels_dev = labels[split_1 : split_2]
	labels_test = labels[split_2:]

	assert len(q1_train) == len(q2_train) == len(labels_train)
	assert len(q1_test) == len(q2_test) == len(labels_test)
	assert len(q1_dev) == len(q2_dev) == len(labels_dev)

	#save data to disk
	np.save(open(q1_train_path, 'wb'), q1_train)
	np.save(open(q2_train_path, 'wb'), q2_train)
	np.save(open(q1_dev_path, 'wb'), q1_dev)
	np.save(open(q2_dev_path, 'wb'), q2_dev)
	np.save(open(q1_test_path, 'wb'), q1_test)
	np.save(open(q2_test_path, 'wb'), q2_test)
	np.save(open(labels_train_path, 'wb'), labels_train)
	np.save(open(labels_dev_path, 'wb'), labels_dev)
	np.save(open(labels_test_path, 'wb'), labels_test)
	np.save(open(word_embeddings_path, 'wb'), word_embeddings)

	#save dataset info
	dataset_info = {
		'train_size' : len(q1_train),
		'dev_size': len(q1_dev),
		'test_size': len(q1_test),
		'vocab_size': vocab_size,
		'embed_dim' : embed_dim,
		'seq_len' : max_seq
	};
	save_json(dataset_info)
	print("Finished preprocessing data and saved")
	return q1_train, q2_train, q1_dev, q2_dev, q1_test, q2_test, labels_train, labels_test, labels_dev, word_embeddings

def data_processed():
	'''
	Returns True if all train, dev, and test datasets, and word embeddings are on the disk
	'''
	for file in [q1_train_path, q2_train_path, q1_dev_path, q2_dev_path, q1_test_path, q2_test_path, labels_train_path, labels_dev_path, labels_test_path, word_embeddings_path]:
		if not os.path.exists(file):
			return False
	return True

def load_dataset():
	if  data_processed():
		q1_train = np.load(open(q1_train_path, 'rb'))
		q2_train = np.load(open(q2_train_path, 'rb'))
		q1_dev = np.load(open(q1_dev_path, 'rb'))
		q2_dev = np.load(open(q2_dev_path, 'rb'))
		q1_test = np.load(open(q1_test_path, 'rb'))
		q2_test = np.load(open(q2_test_path, 'rb'))

		labels_train = np.load(open(labels_train_path, 'rb'))
		labels_test =np.load(open(labels_test_path, 'rb'))
		labels_dev = np.load(open(labels_dev_path, 'rb'))
		word_embeddings = np.load(open(word_embeddings_path, 'rb'))

		return q1_train, q2_train, q1_dev, q2_dev, q1_test, q2_test, labels_train, labels_test, labels_dev, word_embeddings
	else:
		'''
		Be sure to download the dataset and glvoe embeddings before processing anything.
		'''
		# if not os.path.exists(quora_dataset):
		# 	response = requests.get(quora_url, allow_redirects=True)
		# 	with open(quora_dataset, 'wb') as f:
		# 		for data in tqdm(response.iter_content()):
		# 			f.write(data)

		# if not os.path.exists(glove_embeddings):
		# 	response = urlopen(glove_url)
		# 	zipfile = ZipFile(BytesIO(response.read()))
		# 	zipfile.extractall("data/glove")

		return process_dataset()

if __name__ == '__main__':
	load_dataset()
	

