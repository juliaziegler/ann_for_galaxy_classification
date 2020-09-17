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
epochs=30 # training will be repeated 30 times
validation_split=0.1 # it will be trained on 0.9% of training data, 0.1% is put aside for validation

# optimizer which implements the Adam algorithm (https://keras.io/api/optimizers/adam/)
optimizer=tf.keras.optimizers.Adam(lr=learning_rate) 
# metrics 
acc=tf.keras.metrics.BinaryAccuracy(name='accuracy') 
auc=tf.keras.metrics.AUC(multi_label=True, name='area_under_curve')
pre=tf.keras.metrics.Precision(name='precision') 
rec=tf.keras.metrics.Recall(name='recall') 
tp=tf.keras.metrics.TruePositives(name='true_positives')
tn=tf.keras.metrics.TrueNegatives(name='true_negatives')
fp=tf.keras.metrics.FalsePositives(name='false_positives')
fn=tf.keras.metrics.FalseNegatives(name='false_negatives')
metrics=[acc, auc, pre, rec, tp, tn, fp, fn]
input_shape=(1440 ,1)
min_wavelen=5000
max_wavelen=6500
SPLICE_CODE=512 # this code is a placeholder, we will use it only here, corresponds to class 1 (positives)
OTHER_CODE=0 # this is the code for spectra with no comment, corresponds to class 0 (negatives)
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
		# for conv layer: kernel_size is size of convolution window
		# for conv layer: filters is number of 'copies' of the output  
		#   (we can identify as many characteristics as we have filters)
		#   (each filter uses a kernel with different weights)
		# for pooling: max value out of four(=pool_size) is used, other values are dropped
		# for dropout: randomly sets input units to 0 with dropout rate at each step, 
		#   prevents overfitting
		self.kernel_size=3
		self.pool_size=4 
		self.dropout=0.3
		self.model=tf.keras.Sequential(
			[tf.keras.Input(shape=input_shape[0]),
			tf.keras.layers.Reshape((int(input_shape[0]),1)),
			tf.keras.layers.Conv1D(filters=8, kernel_size=self.kernel_size,
						activation='relu'),
			tf.keras.layers.MaxPooling1D(pool_size=self.pool_size),
			tf.keras.layers.Conv1D(filters=16, kernel_size=self.kernel_size,
						activation='relu'),
			tf.keras.layers.MaxPooling1D(pool_size=self.pool_size),
			tf.keras.layers.Conv1D(filters=32, kernel_size=self.kernel_size,
						activation='relu'),
			tf.keras.layers.MaxPooling1D(pool_size=self.pool_size),
			tf.keras.layers.Conv1D(filters=64, kernel_size=self.kernel_size, 
						activation='relu'),
			tf.keras.layers.MaxPooling1D(pool_size=self.pool_size),
			tf.keras.layers.Flatten(),
			tf.keras.layers.Dropout(self.dropout),
			tf.keras.layers.Dense(200, activation='relu'),
			tf.keras.layers.Dense(20, activation='relu'),
			tf.keras.layers.Dropout(self.dropout),
			tf.keras.layers.Dense(1, activation='sigmoid')])
		# compile model
		self.model.compile(optimizer=self.optimizer, loss='categorical_crossentropy',
				    metrics=self.metrics)
		# callback to stop training when validation loss does not improve further
		self.callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=1e-2,
								 patience=5, verbose=1,)]		
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
		hist=pd.DataFrame(history.history) # pandas table with history of training
		loss=hist['loss'] # array with loss for each epoch
		val_loss=hist['val_loss'] # validation loss
		acc=hist['accuracy']
		val_acc=hist['val_accuracy']
		auc=hist['area_under_curve']
		val_auc=hist['val_area_under_curve']
		# precision and recall
		pre=hist['precision']
		val_pre=hist['val_precision']
		rec=hist['recall']
		val_rec=hist['val_recall']
		f1=2*(pre*rec)/(pre+rec)
		val_f1=2*(val_pre*val_rec)/(val_pre+val_rec)
		# TP, TN, FP, FN
		tp=hist['true_positives']
		tn=hist['true_negatives']
		fp=hist['false_positives']
		fn=hist['false_negatives']
		val_tp=hist['val_true_positives']
		val_tn=hist['val_true_negatives']
		val_fp=hist['val_false_positives']
		val_fn=hist['val_false_negatives']
		# test scores
		test_loss=score[0]
		test_acc=score[1]
		test_auc=score[2]
		test_pre=score[3]
		test_rec=score[4]
		test_tp=score[5]
		test_tn=score[6]
		test_fp=score[7]
		test_fn=score[8]
		if test_pre==0 and test_rec==0:
			test_f1=0
		else:
			test_f1=2*(test_pre*test_rec)/(test_pre+test_rec)
		# plot values above
		plt.figure(figsize=(16,11))
		plt.plot(epochs, loss, 'b.', label='loss')
		plt.plot(epochs, val_loss, 'g.', label='validation loss')
		plt.plot(epochs, f1, 'bo', label='F1 score')
		plt.plot(epochs, val_f1, 'go', label='validation F1 score')
		plt.plot(epochs, auc, 'bs', label='AUC')
		plt.plot(epochs, val_auc, 'gs', label='validation AUC')
		plt.xlabel('Epoch')
		plt.ylabel('Score')
		plt.title('Loss Curve for Problem: Bad Splicing'+ ', learning rate='+ 
			   str(learning_rate)+ ', batch size='+ str(batch_size)+ 
			   ', validation split='+ str(validation_split)+
			   '\n kernel size: '+str(self.kernel_size)+', '+
			   ' drop out rate: '+str(self.dropout)+
			   '\n test F1 score: ' + str(test_f1)+
			   '\n test AUC: ' + str(test_auc)+
			   '\n trained on class ratios: 1/2, 1/2'+
			   ', tested on class ratios: 0.91, 0.09')
		plt.legend()
		plt.savefig('Documents/task_512/Losscurve_binary_2_'+ 
			     str(learning_rate) + str(batch_size) + str(validation_split)+ 
			     str(epochs[-1]) +str(self.pool_size)+
			     str(self.kernel_size)+
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
  
  
  
  
