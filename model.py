
import numpy as np
import tensorflow as tf

from parser import JAVADOC_FILE_NAME, METHOD_NAME_FILE_NAME
from parser import METHOD_API_FILE_NAME, METHOD_TOKENS_FILE_NAME

class Model:


	def __init__(self):
		# Intialize some hyperparameters
		self.train_frac = 0.8
		self.valid_frac = 0.2
		self.step_size = 0.01
		self.margin = 0.05


	def _load_data_file(self, file_name):
		dataset = []
		with open(file_name, 'r') as file:
			for line in file:
				line = line.strip()
				dataset.append(line.split())
		return np.array(dataset)

	def train(self, base_dir="data"):
		method_names = self._load_data_file(base_dir + "/" + METHOD_NAME_FILE_NAME)
		method_api_calls = self._load_data_file(base_dir + "/" + METHOD_API_FILE_NAME)
		method_tokens = self._load_data_file(base_dir + "/" + METHOD_TOKENS_FILE_NAME)
		javadoc = self._load_data_file(base_dir + "/" + JAVADOC_FILE_NAME)

		assert len(method_names) == len(method_api_calls)
		assert len(method_tokens) == len(javadoc)
		assert len(method_names) == len(javadoc)
