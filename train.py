import sys
import argparse
import os
import json
import importlib
import matplotlib.pyplot as plt
from utils import *
from preprocess import load_dataset
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau


experiments_dir = "experiments"
json_file = "training_parameters.json"
dataset_params_path = "data/dataset_params.json"

def train_and_evaluate(model, config, q1_train, q2_train, q1_test, q2_test, labels_train, labels_test):
	if config['data_small']:
		q1_train = q1_train[:1000]
		q2_train = q2_train[:1000]
		labels_train = labels_train[:1000]

	checkpoint = ModelCheckpoint(filepath=os.path.join(config['model_dir'], "base_model-{epoch:02d}-{val_loss:.2f}-{acc:.2f}.hdf5"), save_best_only=True)
	# reduce_lr = ReduceLROnPlateau(verbose = 1, patience=10)
	history = model.fit([q1_train, q2_train], labels_train, epochs = config['epochs'], verbose=1, batch_size=config['batch_size'], callbacks=[checkpoint], validation_split=0.1)
	
	#plot
	history_dict = history.history
	acc = history_dict['acc']
	val_acc = history_dict['val_acc']
	loss = history_dict['loss']
	val_loss = history_dict['val_loss']

	epochs = range(1, len(acc) + 1)
	plt.plot(epochs, loss, 'bo', label='Training loss')
	plt.plot(epochs, val_loss, 'b', label='Validatin loss')
	plt.xlabel('Epochs')
	plt.ylabel('Loss')
	plt.legend()
	plt.save_fig(os.path.join(config['model_dir'], 'loss.png'))

	plt.clf()
	plt.plot(epochs, acc, 'bo', label='Training acc')
	plt.plot(epochs, val_acc, 'b', label='Validation acc')
	plt.title('Training and validation accuracy')
	plt.xlabel('Epochs')
	plt.ylabel('Accuracy')
	plt.legend()
	plt.save_fig(os.path.join(config['model_dir'], 'accuracy.png'))


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--model_dir', help='Directory with training parameters,', default='experiments/base_model')
	parser.add_argument('--model', help='Python file specifying the model to train.', default = 'model/base_model.py')
	parser.add_argument('--data_small', help='train using a smaller dataset?', dest='data_small', action='store_true')
	args = parser.parse_args()

	assert os.path.exists(args.model_dir), "directory does not exist : {}".format(args.model_dir)
	assert os.path.exists(args.model), "model file does not exist: {}".format(args.model)

	#load the parameters 
	json_path = os.path.join(args.model_dir, json_file)
	assert os.path.exists(json_path), "no model parameters found at {}".format(args.model_dir)

	#load the parameters from training data
	assert os.path.exists(dataset_params_path)
	config  = read_json(json_path, dataset_params_path)
	config['model_dir'] = args.model_dir
	config['model'] = get_basename(args.model)
	config['data_small'] = args.data_small

	#import model module
	module = importlib.import_module("model.%s" % config['model'])
	
	#load data
	q1_train, q2_train, q1_test, q2_test, labels_train, labels_test, word_embeddings = load_dataset()
	model = module.model(config, word_embeddings)
	
	train_and_evaluate(model, config, q1_train, q2_train, q1_test, q2_test, labels_train, labels_test)
















	















