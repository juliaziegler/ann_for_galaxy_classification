# import libraries
import astropy # library with useful tools often needed in astrophysics
from astropy.io import fits # fits is the data format used here (most common in astrophysics)
import matplotlib.pyplot as plt # makes plots
import numpy as np # library with useful commands for arrays 
import tensorflow as tf # library for machine learning
import pandas as pd # library to manage data tables
import glob # to create list of file names
import math
import os
import pickle # to save data sets
from scipy.interpolate import interp1d
from astropy.convolution import Gaussian1DKernel, Box1DKernel, interpolate_replace_nans, convolve
##############################################################################################
# class 0: no bad splicing, class 1: bad splicing
# define hyperparameters
learning_rate=0.0001 # https://arxiv.org/abs/1412.6980 suggest 0.001 but this seemed a bit too hight for our problem
batch_size=10 # after 10 examples the weights will be updated
epochs=20 # training will be repeated 20 times
validation_split=0.1 # it will be trained on 0.9% of training data, 0.1% is put aside for validation

# optimizer which implements the Adam algorithm (https://keras.io/api/optimizers/adam/)
optimizer=tf.keras.optimizers.Adam(lr=learning_rate) 
# metrics 
cac=tf.keras.metrics.CategoricalAccuracy(name='categorical_accuracy') 
pre_0=tf.keras.metrics.Precision(class_id=0 , name='precision_0') 
rec_0=tf.keras.metrics.Recall(class_id=0 , name='recall_0') 
pre_1=tf.keras.metrics.Precision(class_id=1 , name='precision_1') 
rec_1=tf.keras.metrics.Recall(class_id=1 , name='recall_1') 
metrics=[cac, rec_0, pre_0, rec_1, pre_1]
input_shape=(1440 ,1)
min_wavelen=5000
max_wavelen=6500
SPLICE_CODE=512 # this code is a placeholder, we will use it only here, corresponds to class 1
OTHER_CODE=0 # this is the code for spectra with no comment, corresponds to class 0
FILEPATH_DATA='Documents/task_512/pickle/pickle_512' # filepath where the spec_data is saved
FILEPATH_MODEL='Documents/task_512/models/MODEL_512' # filepath where the models is saved
##############################################################################################
# function to load previously saved training and test data sets 
def load_data_sets(filepath):
	with open(filepath, 'rb') as pkl_in:
		spec_data_train = pickle.load(pkl_in)
		labels_train = pickle.load(pkl_in)
		train_ind = pickle.load(pkl_in)
		spec_data_test = pickle.load(pkl_in)
		labels_test = pickle.load(pkl_in)
		test_ind = pickle.load(pkl_in)
	return spec_data_train, labels_train, spec_data_test, labels_test, train_ind, test_ind
##############################################################################################
# class for neural network
class NN:
	# initialisation of the neural network 
	def __init__(self, optimizer, metrics):
		self.optimizer=optimizer
		self.metrics=metrics
	def build_model(self, input_shape):
		# model is sequential: straightforward, limited to single-input, 
		#   single-output stacks of layers
		# example taken from https://keras.io/examples/vision/mnist_convnet/ and adapted
		# for conv layer: kernel_size is size of convolution window
		# for conv layer: filters is number of 'copies' of the output  
		#   (we can identify as many characteristics as we have filters)
		#   (each filter uses a kernel with different weights)
		# for pooling: max value out of four(=pool_size) is used, other values are dropped
		# for dropout: randomly sets input units to 0 with dropout rate at each step, 
		#   prevents overfitting
		self.kernel_size_1=100
		self.kernel_size_2=100
		self.pool_size_2=4 
		self.pool_size_3=4
		self.pool_size_4=4 
		self.dropout=0.6
		self.model=tf.keras.Sequential(
			[tf.keras.Input(shape=input_shape[0]),
			tf.keras.layers.Reshape((int(input_shape[0]),1)),
			tf.keras.layers.Conv1D(filters=16, kernel_size=self.kernel_size_1,
						activation='relu'),
			tf.keras.layers.MaxPooling1D(pool_size=self.pool_size),
			tf.keras.layers.Conv1D(filters=32, kernel_size=self.kernel_size_2,
						activation='relu'),
			tf.keras.layers.MaxPooling1D(pool_size=self.pool_size),
			tf.keras.layers.Flatten(),
			tf.keras.layers.Dropout(self.dropout),
			tf.keras.layers.Dense(700, activation='relu'),
			tf.keras.layers.Dense(500, activation='relu'),
			tf.keras.layers.Dense(50, activation='relu'),
			tf.keras.layers.Dropout(self.dropout),
			tf.keras.layers.Dense(2, activation='softmax')])
		# compile model
		self.model.compile(optimizer=self.optimizer, loss='categorical_crossentropy',
				    metrics=self.metrics)
		return self.model
	def train(self, input_data, labels, batch_size, epochs, validation_split):
		history=self.model.fit(x=input_data, y=labels, batch_size=batch_size, epochs=epochs,
					validation_split=validation_split)
		return history
	def evaluate(self, input_data_test, labels_test):
		score = self.model.evaluate(input_data_test, labels_test)
		return score
	def predict(self, input_data):
		prediction=self.model.predict(input_data, batch_size=None)
		return prediction
	def save_model(self, filepath):
		tf.keras.models.save_model(self.model, filepath, overwrite=True,
					    include_optimizer=True, save_format='tf')
	def plot_loss(self, history, score):
		# plot the loss curve
		epochs=history.epoch # array with numbers of epochs
		hist=pd.DataFrame(history.history) # pandas table with history of the fitting
		loss=hist['loss'] # array with loss for each epoch
		val_loss=hist['val_loss'] # validation loss
		caccuracy=hist['categorical_accuracy']
		val_caccuracy=hist['val_categorical_accuracy']
		# recall and precision for class 0 (neither fringed nor dwarf)
		recall_0=hist['recall_0']
		val_recall_0=hist['val_recall_0']
		precision_0=hist['precision_0']
		val_precision_0=hist['val_precision_0']
		f1_0=2*(precision_0*recall_0)/(precision_0+recall_0)
		val_f1_0=2*(val_precision_0*val_recall_0)/(val_precision_0+val_recall_0)
		# recall and precision for class 1 (fringed)
		recall_1=hist['recall_1']
		val_recall_1=hist['val_recall_1']
		precision_1=hist['precision_1']
		val_precision_1=hist['val_precision_1']
		f1_1=2*(precision_1*recall_1)/(precision_1+recall_1)
		val_f1_1=2*(val_precision_1*val_recall_1)/(val_precision_1+val_recall_1)
		# test score
		test_f1_0=2*(score[2]*score[3])/(score[2]+score[3])
		test_f1_1=2*(score[4]*score[5])/(score[4]+score[5])
		# plot all of the above but leave out first epoch (loss is too high)
		plt.figure(figsize=(16,11))
		plt.plot(epochs, loss, 'b.', label='loss')
		plt.plot(epochs, val_loss, 'g.', label='validation loss')
		plt.plot(epochs, f1_0, 'bo', label='F1 score class 0')
		plt.plot(epochs, val_f1_0, 'go', label='validation F1 score class 0')
		plt.plot(epochs, f1_1, 'bs', label='F1 score class 1')
		plt.plot(epochs, val_f1_1, 'gs', label='validation F1 score class 1')
		plt.xlabel('Epoch')
		plt.ylabel(' ')
		plt.title('Loss Curve for Problem: Bad Splicing'+ ', learning rate='+ 
			   str(learning_rate)+ ', batch size='+ str(batch_size)+ 
			   ', validation split='+ str(validation_split)+
			   ' drop out rate: '+str(self.dropout)+
			   '\n test F1 score class 0: ' + str(test_f1_0)+
			   '\n test F1 score class 1: ' + str(test_f1_1)+
			   '\n trained on class ratios: 1/2, 1/2'+
			   ', tested on class ratios: 0.91, 0.09')
		plt.legend()
		plt.savefig('Documents/task_512/Losscurve_'+ 
			     str(learning_rate) + str(batch_size) + str(validation_split)+ 
			     str(epochs[-1]) +str(self.pool_size)+
			     str(self.kernel_size_1)+str(self.kernel_size_2)+ 
			     str(self.dropout) +'.png')
		plt.show(block=False)
		plt.close()
		return
##############################################################################################
if __name__ == "__main__":
  # load data set
  spec_data_train, labels_train, spec_data_test, labels_test, train_ind, test_ind = load_data_sets(FILEPATH_DATA)
  # now create nn and train on this set
  nn=NN(optimizer, metrics)
  model=nn.build_model(input_shape)
  history=nn.train(spec_data_train, labels_train, batch_size, epochs, validation_split)
  score=nn.evaluate(spec_data_test, labels_test)
  nn.plot_loss(history, score)
  # save the trained model 
  nn.save_model(FILEPATH_MODEL)    
  
  
  
  
