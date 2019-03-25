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
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, Callback
import shutil

# experiments = "experiments"
dataset_params_path = "data/dataset_params.json"
	
def train_and_evaluate(model, config, q1_train, q2_train, q1_test, q2_test, labels_train, labels_test):
	if config['train_subset']:
		q1_train = q1_train[:1000]
		q2_train = q2_train[:1000]
		labels_train = labels_train[:1000]
	callbacks = []
	if config['save']:
		checkpoint = ModelCheckpoint(filepath=os.path.join(config['model_dir'], "{epoch:02d}-{val_loss:.2f}-{acc:.2f}.hdf5"), save_best_only=True)
		callbacks.append(checkpoint)	
	# reduce_lr = ReduceLROnPlateau(verbose = 1, patience=10)

	if config['weights'] is not None:
		model.load_weights(config['weights'])
		initial_epoch = get_initial_epoch(config['weights'])
	else:
		initial_epoch = 0
	history = model.fit([q1_train, q2_train], labels_train, epochs = config['epochs'], initial_epoch = initial_epoch, verbose=1, batch_size=config['batch_size'], callbacks=callbacks, validation_split=0.1)
	
	# save model weights and specification
	if config['save']:
		weights_dest = os.path.join(config['model_dir'], '{}.hdf5'.format(config['model']))
		model_dest =  os.path.join(config['model_dir'], '{}.json'.format(config['model']))
		model.save_weights(weights_dest)
		model_json = model.to_json()
		with open(model_dest, "w") as f:
			f.write(model_json)
	
	history_dict = history.history 
	np.save(open(os.path.join(config['model_dir'], "history_dict.npy"), "wb"), history_dict)
	acc = history_dict['acc']
	val_acc = history_dict['val_acc']
	loss = history_dict['loss']
	val_loss = history_dict['val_loss']

	val_f1_score = history_dict['val_f1_score']
	f1_score = history_dict['f1_score']

	val_precision = history_dict['val_precision']
	precision = history_dict['precision']

	recall = history_dict['recall']
	val_recall = history_dict['val_recall']

	epochs = range(1, len(acc) + 1)
	plt.plot(epochs, loss, 'bo', label='loss')
	plt.plot(epochs, val_loss, 'b', label='val loss')
	plt.xlabel('Epochs')
	plt.ylabel('Loss')
	plt.legend()
	plt.savefig(os.path.join(config['model_dir'], 'loss.png'))

	plt.clf()
	plt.plot(epochs, acc, 'bo', label='acc')
	plt.plot(epochs, val_acc, 'b', label='val acc')
	plt.title('Training and validation accuracy')
	plt.xlabel('Epochs')
	plt.ylabel('Accuracy')
	plt.legend()
	plt.savefig(os.path.join(config['model_dir'], 'accuracy.png'))

	plt.clf()
	plt.plot(epochs, f1_score, 'bo', label='F1 score')
	plt.plot(epochs, val_f1_score, 'b', label='val F1 score')
	plt.title('Training and validation F1 score')
	plt.xlabel('Epochs')
	plt.ylabel('F1 score')
	plt.legend()
	plt.savefig(os.path.join(config['model_dir'], 'f1_score.png'))

	plt.clf()
	plt.plot(epochs, precision, 'bo', label='precision')
	plt.plot(epochs, val_precision, 'b', label='val precision')
	plt.title('Training and validation precision')
	plt.xlabel('Epochs')
	plt.ylabel('Precision')
	plt.legend()
	plt.savefig(os.path.join(config['model_dir'], 'precision.png'))

	plt.clf()
	plt.plot(epochs, recall, 'bo', label='recall')
	plt.plot(epochs, val_recall, 'b', label='val recall')
	plt.title('Training and validation recall')
	plt.xlabel('Epochs')
	plt.ylabel('recall')
	plt.legend()
	plt.savefig(os.path.join(config['model_dir'], 'recall.png'))


	dev_stats = {'acc' : max(acc), 'val_acc' : max(val_acc), 'precision': max(precision), 'val_precision': max(val_precision), 'f1_score' : max(f1_score), 'val_f1_score': max(val_f1_score)}
	save_json(dev_stats, os.path.join(config['model_dir'], 'dev_stats.json'))

	# evaluate on test dataset
	if config['train_subset']:
		labels_test = labels_test[:100]
		q2_test = q2_test[:100]
		q1_test = q1_test[:100]

	print('Testing ...')
	loss, accuracy, precision, recall, f1_score = model.evaluate([q1_test, q2_test], labels_test)
	test_stats = {'acc' : accuracy, 'precision': precision, 'recall' : recall, 'f1_score': f1_score}
	print('loss: {0:.4f}, accuracy: {0:.4f} precision: {0:.4f}, recall: {0:.4f}, f1_score: {0:.4f}'.format(loss, accuracy, precision, recall, f1_score))
	save_json(test_stats, os.path.join(config['model_dir'], 'test_stats.json'))

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--save', help='save model after training / checkpoints', dest='save', action='store_true')
	parser.add_argument('--model', help='name of model to train.', default ='base_model')
	parser.add_argument('--model_dir',help='directory with training parameters file / weights', default='experiments/base_model')
	parser.add_argument('--train_subset', help='train using a smaller dataset?', dest='train_subset', action='store_true')
	parser.add_argument('--clean', help='clean preprocess dataset', dest='clean', action='store_true')
	parser.add_argument('--weights', help='.hdf5 file in model_dir to load weights from', default=None)
	args = parser.parse_args()

	# training parameters file
	json_file = "training_parameters.json"
	training_parameters = os.path.join(args.model_dir, json_file)
	assert os.path.exists(training_parameters)

	#load the parameters from training data
	assert os.path.exists(dataset_params_path)
	config  = read_json(training_parameters, dataset_params_path)
	config['model_dir'] = args.model_dir
	config['model'] = args.model
	config['train_subset'] = args.train_subset
	config['clean'] = args.clean
	config['weights'] = None
	config['save'] = args.save
	if args.weights is not None:
		weights = os.path.join(args.model_dir, args.weights)
		assert os.path.exists(weights)
		config['weights'] = weights

	'''
	training parameters are located in config['model_dir'] = experiments/model_name/job_category/job_name
	However, if config['model_dir'] = experiments/model_name, we need to create directories to maintain uniformity in how we structure our training -> where to get training parameters/other files
	'''
	if get_tokens_in_filename(config['model_dir']) == 2:
		config['model_dir'] = create_model_dir(config['model_dir'], config['train_subset'])
		shutil.copy(training_parameters, config['model_dir'])

	#import model module
	module = importlib.import_module("model.%s" % config['model'])
	
	#load data
	q1_train, q2_train, q1_test, q2_test, labels_train, labels_test, word_embeddings = load_dataset(config)
	model = module.model(config, word_embeddings)
	
	print("starting training for {model}, epochs = {epochs}".format(model=config['model'], epochs=config['epochs']))
	train_and_evaluate(model, config, q1_train, q2_train, q1_test, q2_test, labels_train, labels_test)


	
