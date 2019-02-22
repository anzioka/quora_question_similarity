import sys
import argparse
import os
import json
import importlib
from preprocess import load_dataset
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

experiments_dir = "experiments"
json_file = "training_parameters.json"
dataset_params_path = "data/dataset_params.json"

class Config(object):
	"""docstring for Config"""
	def __init__(self, *pathnames):
		#Read data from json file and stores the the data in keys : value format
		# assert os.path.exists(pathname), "File does not exist: {}".format(pathname)
		for path in pathnames:
			with open(path) as f:
				data = json.load(f)
				for k, v in data.items():
					setattr(self, k, v)



def train_and_evaluate(model, config, q1_train, q2_train, q1_test, 
	q2_test, labels_train, labels_test):

	checkpoint = ModelCheckpoint(filepath=os.path.join(config.model_dir, config.model + "-{epoch:02d}-{val_loss:.2f}-{acc:.2f}.hdf5"), save_best_only=True)
	history = model.fit([q1_train, q2_train], labels_train, epochs = config.epochs, verbose=1, batch_size=config.batch_size, callbacks=[checkpoint], validation_split=0.1)

	return history;

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--model_dir', help='Directory with training parameters,', default='experiments/base_model')
	parser.add_argument('--model', help='Python file in specifying the model to train.', default = 'model/base_model.py')
	args = parser.parse_args()

	assert os.path.exists(args.model_dir), "directory does not exist : {}".format(args.model_dir)
	assert os.path.exists(args.model), "model file does not exist: {}".format(args.model)

	#load the parameters 
	json_path = os.path.join(args.model_dir, json_file)
	assert os.path.exists(json_path), "no model parameters found at {}".format(args.model_dir)

	#load the parameters from training data
	assert os.path.exists(dataset_params_path)
	config  = Config(json_path, dataset_params_path)
	config.model_dir = args.model_dir
	config.model = os.path.splitext(os.path.basename(args.model))[0]

	#import model module
	module = importlib.import_module("model.%s" % config.model)
	
	#load data
	q1_train, q2_train, q1_test, q2_test, labels_train, labels_test, word_embeddings = load_dataset()
	model = module.model(config, word_embeddings)
	
	train_and_evaluate(model, config, q1_train, q2_train, q1_test, q2_test, labels_train, labels_test)
















	















