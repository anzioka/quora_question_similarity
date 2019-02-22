
import os
import random
import requests
import numpy as np
import json
import csv
import re
import pandas as pd
from tqdm import tqdm
from keras.utils.data_utils import get_file
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split 
from zipfile import ZipFile
from io import BytesIO
from urllib.request import urlopen
from collections import Counter

glove_url = 'http://nlp.stanford.edu/data/glove.840B.300d.zip'
quora_url = 'http://qim.fs.quoracdn.net/quora_duplicate_questions.tsv'
glove_embeddings = 'data/glove/glove.840B.300d.txt'
quora_dataset = 'data/quora/quora_duplicate_questions.tsv'
word_embeddings_path = "data/embeddings/embeddings.npy"
q1_train_path= "data/train/q1_train.npy"
q2_train_path = "data/train/q2_train.npy"
q1_test_path = "data/test/q1_test.npy"
q2_test_path = "data/test/q2_test.npy"
labels_train_path = "data/labels/labels_train.npy"
labels_test_path = "data/labels/labels_test.npy"
dataset_params = "data/dataset_params.json"
embed_dim = 300

def save_json(dataset_info):
	'''saves file to json 
	'''
	with open(dataset_params, 'w') as f:
		d = {k : v for k, v in dataset_info.items()}
		json.dump(d, f, indent=4)

def tokenize(sequence):
	return [i.strip().lower() for i in re.split('\W+', sequence) if i.strip()]
def build_vocab(data):
	counter = Counter()

	for sent in tqdm(data):
		for word in tokenize(sent):
			counter[word] += 1
	words = [wordcount[0] for wordcount in counter.most_common()]
	word_index = {w: i + 1 for i, w in enumerate(words)}
	return word_index

def texts_to_sequences(q1_data, q2_data, labels, word_index):
	q_1 = []
	q_2 = []
	y = []
	for sent1, sent2, label in tqdm(zip(q1_data, q2_data, labels)):
		q_1.append([word_index[w] for w in tokenize(sent1)])
		q_2.append([word_index[w] for w in tokenize(sent2)])
		y.append((np.array([0, 1]) if label == '1' else np.array([1, 0])))
	y = np.array(y)
	return q_1, q_2, y

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

	print("Building vocabulary...")
	word_index = build_vocab(q1_data + q2_data)
	vocab_size = len(word_index) + 1

	print ("vocab_size: {}".format(vocab_size))

	#convert text to sequence of integers
	print ("converting text to integers")
	q_1, q_2, labels = texts_to_sequences(q1_data, q2_data, labels, word_index)

	max_seq = max([len(i) for i in q_1] + [len(i) for i in q_2])
	print ("Maximum sequence in q_1: {}".format(max_seq))

	q_1 = pad_sequences(q_1, maxlen = max_seq, padding="post")
	q_2 = pad_sequences(q_2, maxlen = max_seq, padding="post")
	assert q_2.shape == q_1.shape

	#split into train and test sets
	q1_train, q1_test, q2_train, q2_test, labels_train, labels_test = train_test_split(q_1, q_2, labels, test_size=0.1, random_state=42)
	print ("q1_train: {}, q2_train shape: {}, q1_test: {}, q2_test: {}, labels_train: {}, labels_test: {}".format(q1_train.shape, q2_train.shape, q1_test.shape, q2_test.shape, labels_train.shape, labels_test.shape))

	#create embedding dictionary
	print ("Creating embedding dictionary...")
	embedding_dict = {}
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
		else:
			word_embeddings[i] = np.random.uniform(-0.25, 0.25, embed_dim)
	
	print ("Shape of word word_embeddings: " + str(word_embeddings.shape))

	#save data to disk
	np.save(open(q1_train_path, 'wb'), q1_train)
	np.save(open(q2_train_path, 'wb'), q2_train)
	np.save(open(q1_test_path, 'wb'), q1_test)
	np.save(open(q2_test_path, 'wb'), q2_test)
	np.save(open(labels_train_path, 'wb'), labels_train)
	np.save(open(labels_test_path, 'wb'), labels_test)
	np.save(open(word_embeddings_path, 'wb'), word_embeddings)

	#save dataset info
	dataset_info = {
		'train_size' : len(q1_train),
		'test_size': len(q1_test),
		'vocab_size': vocab_size,
		'embed_dim' : embed_dim,
		'seq_len' : max_seq
	};
	save_json(dataset_info)
	print("Finished preprocessing data and saved")
	return q1_train, q2_train, q1_test, q2_test, labels_train, labels_test, word_embeddings

def data_processed():
	'''
	Returns True if all train, dev, and test datasets, and word embeddings are on the disk
	'''
	for file in [q1_train_path, q2_train_path, q1_test_path, q2_test_path, labels_train_path, labels_test_path, word_embeddings_path]:
		if not os.path.exists(file):
			return False
	return True 

def load_dataset():
	if  data_processed():
		q1_train = np.load(open(q1_train_path, 'rb'))
		q2_train = np.load(open(q2_train_path, 'rb'))
		q1_test = np.load(open(q1_test_path, 'rb'))
		q2_test = np.load(open(q2_test_path, 'rb'))

		labels_train = np.load(open(labels_train_path, 'rb'))
		labels_test =np.load(open(labels_test_path, 'rb'))
		word_embeddings = np.load(open(word_embeddings_path, 'rb'))

		return q1_train, q2_train, q1_test, q2_test, labels_train, labels_test, word_embeddings
	else:
		'''
		Be sure to download the dataset and glvoe embeddings before processing anything.
		'''
		if not os.path.exists(quora_dataset):
			response = requests.get(quora_url, allow_redirects=True)
			with open(quora_dataset, 'wb') as f:
				for data in tqdm(response.iter_content()):
					f.write(data)

		if not os.path.exists(glove_embeddings):
			response = urlopen(glove_url)
			zipfile = ZipFile(BytesIO(response.read()))
			zipfile.extractall("data/glove")

		return process_dataset()

if __name__ == '__main__':
	load_dataset()
	

