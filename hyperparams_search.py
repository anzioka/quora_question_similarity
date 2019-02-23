import argparse
import os
from subprocess import check_call
import sys

parameters = "training_parameters.json"

parser = argparse.ArgumentParser()
parser.add_argument('--model', help='python file specifying the model to train.', default = 'model/base_model.py')
parser.add_argument('--model_dir', help='directory with training parameters,', default='experiments/base_model')
parameters.add_argument('--parameter', help='parameter to tune', default='learning_rate')

def launch_training_job(parent_dir, task, model):
	pass

def searchLearningRate():
	pass


if __name__ == '__main__':
	args = parser.parse_args()
	parameters = os.path.join(args.model_dir, parameters)
	assert os.path.exists(parameters), "No json config file found at {}".format(args.model_dir)
	assert os.path.exists(args.model), 'No model file found at {}'.format(args.model)

	if args.parameter == 'learning_rate':
		searchLearningRate()





