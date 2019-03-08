import argparse
import os
from subprocess import check_call
import sys
import numpy as np
import utils
import operator

parameters = "training_parameters.json"
PYTHON = sys.executable

parser = argparse.ArgumentParser()
parser.add_argument('--model', help='python file specifying the model to train.', default = 'base_model')
parser.add_argument('--model_dir',help='directory with training parameters file / weights', default='experiments/base_model')


def launch_training_job(model_dir, model, params):
	# model_dir contains training parameters
	# need to copy over the params to the model_dir
	params_file = os.path.join(model_dir, parameters)
	utils.save_json(params, params_file)

	#launch training job with config
	cmd = "{python} train.py --model_dir {model_dir} --train_subset --model {model}".format(python=PYTHON, model_dir = model_dir, model = model)
	print (cmd)
	check_call(cmd, shell=True)


def max_len(model):
	pass

def hidden_size(model):
	pass
def l2_regularization(model, start, end, count):
	parent_dir = os.path.join(args.model_dir, "l2")
	os.makedirs(parent_dir, exist_ok=True)
	values = np.logspace(start,end,count)
	result = {}
	for x in values:
		params['lr'] = x
		model_dir = os.path.join(parent_dir, str(x))
		os.makedirs(model_dir, exist_ok=True)
		launch_training_job(model_dir, model, params)

		f = os.path.join(model_dir, "history_dict.npy")
		d = np.load(f).item()

		max_val = max(d.items(), key = operator.itemgetter(1))
		result[x] = max_val[1]

	print("validation accuracy values with l2_regularization"):
	print(result)

if __name__ == '__main__':
	args = parser.parse_args()
	params_file = os.path.join(args.model_dir, parameters)
	assert os.path.exists(params_file)
	params = utils.read_json(params_file)
	l2_regularization('base_model', -4, -2, 10)
	# max_len('base_model')
	# hidden_size('base_model')






