### YOUR CODE HERE
# import tensorflow as tf
# import torch

import os, argparse
import numpy as np
from Model import MyModel
from DataLoader import load_data, train_valid_split,load_testing_images_private_data
from Configure import model_configs, training_configs,configure
from ImageUtils import visualize


# parser = argparse.ArgumentParser()
# parser.add_argument("mode", help="train, test or predict")
# parser.add_argument("data_dir", help="path to the data")
# parser.add_argument("--save_dir", help="path to save the results")
# args = parser.parse_args()

if __name__ == '__main__':
	os.environ["CUDA_VISIBLE_DEVICES"]='-1'
	model_and_run_configs = configure()
	model = MyModel(model_and_run_configs)
	print(model.network)
	print('----------------')
	os.makedirs(model_and_run_configs.save_dir,exist_ok=True)

	if model_and_run_configs.mode == 'train':
		x_train, y_train, x_test, y_test = load_data(model_and_run_configs.data_dir)
		x_train, y_train, x_valid, y_valid = train_valid_split(x_train, y_train)

		model.train(x_train, y_train, training_configs, x_valid, y_valid)
		model.evaluate(x_test, y_test)

	elif model_and_run_configs.mode == 'test':
		# Testing on public testing dataset
		_, _, x_test, y_test = load_data(model_configs.data_dir)
		model.evaluate(x_test, y_test,model_and_run_configs.final_model_path)

	elif model_configs.mode == 'predict':
		# Loading private testing dataset
		data_path = os.path.join(model_and_run_configs.data_dir,'private_test_images_2022.npy')
		x_test = load_testing_images_private_data(data_path)
		# visualizing the first testing image to check your image shape
		visualize(x_test[0], 'test.png')
		# Predicting and storing results on private testing dataset

		_,predictions = model.predict_prob(x_test,model_and_run_configs.final_model_path)
		print(predictions.shape)
		np.save(os.path.join(model_and_run_configs.save_dir,"predictions.npy"), predictions)
		

### END CODE HERE

