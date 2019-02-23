import os
import json
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
	assert os.path.exists(filename), 'file {} does not exist'.format(filename)
	with open(filename, 'w') as f:
		item = {k: v for k, v in config.items()}
		json.dump(k, v, indent=4)
	#config is an object containing key: dict values
def get_initial_epoch(checkpoint_path):
	filename = get_basename(checkpoint_path)
	return int(filename.split("-")[1])