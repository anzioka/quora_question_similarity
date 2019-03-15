import os
import json
import shutil

from keras import backend as K

def get_basename(filename):
	return os.path.splitext(os.path.basename(filename))[0]

def read_json(*pathnames):
	config = {}
	for path in pathnames:
		with open(path, "r") as f:
			data = json.load(f)
			for k, v in data.items():
				config[k] = v
	return config

def save_json(config, filename):
	with open(filename, 'w') as f:
		d = {k: v for k, v in config.items()}
		json.dump(d, f, indent=4)
	#config is an object containing key: dict values
def get_initial_epoch(checkpoint_path):
	filename = get_basename(checkpoint_path)
	return int(filename.split("-")[1])

def get_tokens_in_filename(filename):
	return len(filename.split("/"));

def create_model_dir(model_dir, data_subset):
	import time
	time = str(time.time()).split('.')[0]
	if (data_subset):
		target = model_dir + "/subset/" + time
	else:
		target = model_dir + '/complete/' + time
	if os.path.exists(target):
		shutil.rmtree(target)
	os.makedirs(target, exist_ok = True) # mkdir -p
	return target

#custom metrics
def f1_score(y_true, y_pred):
	c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
	c2 = K.sum(K.round(K.clip(y_pred, 0, 1)))
	c3 = K.sum(K.round(K.clip(y_true, 0, 1)))

	# If there are no true samples, fix the F1 score at 0.
	# How many selected items are relevant?
	precision = c1 / (c2 + K.epsilon())
	# How many relevant items are selected?
	recall = c1 / (c3 + K.epsilon())

	# Calculate f1_score
	f1_score = 2 * (precision * recall) / (precision + recall + K.epsilon())
	return f1_score

def precision(y_true, y_pred):
	#count positive samples

	c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
	c2 = K.sum(K.round(K.clip(y_pred, 0, 1)))

	#how many selected items are relevant
	precision = c1 / (c2+K.epsilon())

	return precision

def recall(y_true, y_pred):

	#count positive
	c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
	c3 = K.sum(K.round(K.clip(y_true, 0, 1)))

	#how many relevant
	recall = c1 / (c3+K.epsilon())
	return recall

def exponent_neg_manhattan_distance(left, right):
    return K.exp(-K.sum(K.abs(left-right), axis=1, keepdims=True))



