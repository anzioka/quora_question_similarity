import sys
import argparse
import os
import json
import numpy as np
import importlib
import matplotlib
matplotlib.use('PS')
import matplotlib.pyplot as plt
from utils import *
from preprocess import load_dataset
from keras.models import load_model
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau


# experiments = "experiments"
dataset_params_path = "data/dataset_params.json"

def train_and_evaluate(model, config, q1_train, q2_train, q1_test, q2_test, labels_train, labels_test):
	if config['train_subset']:
		q1_train = q1_train[:1000]
		q2_train = q2_train[:1000]
		labels_train = labels_train[:1000]

	checkpoint = ModelCheckpoint(filepath=os.path.join(config['model_dir'], "{epoch:02d}-{val_loss:.2f}-{acc:.2f}.hdf5"), save_best_only=True)
	# reduce_lr = ReduceLROnPlateau(verbose = 1, patience=10)

	if config['weights'] is not None:
		model.load_weights(config['weights'])
		initial_epoch = get_initial_epoch(config['weights'])
	else:
		initial_epoch = 0
	history = model.fit([q1_train, q2_train], labels_train, epochs = config['epochs'], initial_epoch = initial_epoch, verbose=1, batch_size=config['batch_size'], callbacks=[checkpoint], validation_split=0.1)
	
	#save whole model: architecture, weights and optimizer state
	model.save(os.path.join(config['model_dir'], '{model}.hdf5'.format(model = config['model'])))
	
	history_dict = history.history
	np.save(open(os.path.join(config['model_dir'], "history_dict.npy"), "wb"), history_dict)

	acc = history_dict['acc']
	val_acc = history_dict['val_acc']
	loss = history_dict['loss']
	val_loss = history_dict['val_loss']

	epochs = range(1, len(acc) + 1)
	plt.plot(epochs, loss, 'bo', label='Training loss')
	plt.plot(epochs, val_loss, 'b', label='Validation loss')
	plt.xlabel('Epochs')
	plt.ylabel('Loss')
	plt.legend()
	plt.savefig(os.path.join(config['model_dir'], 'loss.png'))

	plt.clf()
	plt.plot(epochs, acc, 'bo', label='Training acc')
	plt.plot(epochs, val_acc, 'b', label='Validation acc')
	plt.title('Training and validation accuracy')
	plt.xlabel('Epochs')
	plt.ylabel('Accuracy')
	plt.legend()
	plt.savefig(os.path.join(config['model_dir'], 'accuracy.png'))

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--model', help='name of model to train.', default ='base_model')
	parser.add_argument('--model_dir',help='directory with training parameters file / weights', default='experiments/base_model')
	parser.add_argument('--train_subset', help='train using a smaller dataset?', dest='train_subset', action='store_true')
	parser.add_argument('--clean', help='clean preprocess dataset', dest='clean', action='store_true')
	parser.add_argument('--weights', help='.hdf5 file in model_dir to load weights from', default=None)
	args = parser.parse_args()

	# training parameters file
	json_file = "training_parameters.json"
	json_path = os.path.join(args.model_dir, json_file)
	print (json_path)
	assert os.path.exists(json_path)

	#load the parameters from training data
	assert os.path.exists(dataset_params_path)
	config  = read_json(json_path, dataset_params_path)
	config['model_dir'] = args.model_dir
	config['model'] = args.model
	config['train_subset'] = args.train_subset
	config['clean'] = args.clean
	config['weights'] = None
	if args.weights is not None:
		weights = os.path.join(args.model_dir, args.weights)
		assert os.path.exists(weights)
		config['weights'] = weights

	#import model module
	module = importlib.import_module("model.%s" % config['model'])
	
	#load data
	q1_train, q2_train, q1_test, q2_test, labels_train, labels_test, word_embeddings = load_dataset(config)
	model = module.model(config, word_embeddings)
	
	print("starting training for {model}, epochs = {epochs}".format(model=config['model'], epochs=config['epochs']))
	train_and_evaluate(model, config, q1_train, q2_train, q1_test, q2_test, labels_train, labels_test)
















	















