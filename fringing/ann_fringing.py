# import libraries
import matplotlib.pyplot as plt # makes plots
import numpy as np # library with useful commands for arrays 
import tensorflow as tf # library for machine learning
import pandas as pd # library to manage data tables
import pickle # to save data sets
##############################################################################################
# class 0: neither fringed nor m-star(dwarf), class 1: fringed, class 2: m-star(dwarf)
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
pre_2=tf.keras.metrics.Precision(class_id=2 , name='precision_2') 
rec_2=tf.keras.metrics.Recall(class_id=2 , name='recall_2') 
metrics=[cac, rec_0, pre_0, rec_1, pre_1, rec_2, pre_2]
input_shape=(4190 ,1) 
min_wavelen=4400 
max_wavelen=8750
DWARF_CODE=32 # this code is a placeholder, we will use it only here, corresponds to class 2
FRINGED_CODE=64 # this is the code for fringed spectra, corresponds to class 1
OTHER_CODE=0 # this is the code for spectra with no comment, corresponds to class 0
FILEPATH_DATA='Documents/task_11/pickle/pickle64' # filepath where the spec_data is saved
FILEPATH_MODEL='Documents/task_11/models/MODEL64' # filepath where the models is saved
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
		# 	single-output stacks of layers
		# for conv layer: kernel_size is size of convolution window
		# for conv layer: filters is number of 'copies' of the output 
		# 	(we can identify as many characteristics as we have filters)
		# 	(each filter uses a kernel with different weights)
		# for pooling: max value out of four(=pool_size) is used, other values are dropped
		# for dropout: randomly sets input units to 0 with dropout rate at each step, 
		# 	prevents overfitting
		self.pool_size=4
		self.dropout=0.6
		self.kernel_size_0=9
		self.kernel_size_1=27
		self.kernel_size_2=81
		self.kernel_size_3=243
		self.kernel_size_4=729
		# input layer
		inputs = tf.keras.Input(shape=input_shape[0])
		x = tf.keras.layers.Reshape((int(input_shape[0]),1))(inputs)
		# zeroth convolution block
		x_0 = tf.keras.layers.Conv1D(filters=8, kernel_size=self.kernel_size_0,
					     activation='relu')(x)
		x_0 = tf.keras.layers.MaxPooling1D(pool_size=self.pool_size)(x_0)
		x_0 = tf.keras.layers.Conv1D(filters=16, kernel_size=self.kernel_size_0,
						activation='relu')(x_0)
		x_0 = tf.keras.layers.MaxPooling1D(pool_size=self.pool_size)(x_0)
		x_0 = tf.keras.layers.Flatten()(x_0)
		x_0 = tf.keras.layers.Dropout(self.dropout)(x_0)		
		# first convolution block
		x_1 = tf.keras.layers.Conv1D(filters=8, kernel_size=self.kernel_size_1,
					     activation='relu')(x)
		x_1 = tf.keras.layers.MaxPooling1D(pool_size=self.pool_size)(x_1)
		x_1 = tf.keras.layers.Conv1D(filters=16, kernel_size=self.kernel_size_1,
						activation='relu')(x_1)
		x_1 = tf.keras.layers.MaxPooling1D(pool_size=self.pool_size)(x_1)
		x_1 = tf.keras.layers.Flatten()(x_1)
		x_1 = tf.keras.layers.Dropout(self.dropout)(x_1)
		# second convolution block
		x_2 = tf.keras.layers.Conv1D(filters=8, kernel_size=self.kernel_size_2,
					     activation='relu')(x)
		x_2 = tf.keras.layers.MaxPooling1D(pool_size=self.pool_size)(x_2)
		x_2 = tf.keras.layers.Conv1D(filters=16, kernel_size=self.kernel_size_2,
						activation='relu')(x_2)
		x_2 = tf.keras.layers.MaxPooling1D(pool_size=self.pool_size)(x_2)
		x_2 = tf.keras.layers.Flatten()(x_2)
		x_2 = tf.keras.layers.Dropout(self.dropout)(x_2)
		# third convolution block
		x_3 = tf.keras.layers.Conv1D(filters=8, kernel_size=self.kernel_size_3,
					     activation='relu')(x)
		x_3 = tf.keras.layers.MaxPooling1D(pool_size=self.pool_size)(x_3)
		x_3 = tf.keras.layers.Conv1D(filters=16, kernel_size=self.kernel_size_3,
						activation='relu')(x_3)
		x_3 = tf.keras.layers.MaxPooling1D(pool_size=self.pool_size)(x_3)
		x_3 = tf.keras.layers.Flatten()(x_3)
		x_3 = tf.keras.layers.Dropout(self.dropout)(x_3)
		# fourth convolution block
		x_4 = tf.keras.layers.Conv1D(filters=8, kernel_size=self.kernel_size_4,
					     activation='relu')(x)
		x_4 = tf.keras.layers.MaxPooling1D(pool_size=self.pool_size)(x_4)
		x_4 = tf.keras.layers.Conv1D(filters=16, kernel_size=self.kernel_size_4,
						activation='relu')(x_4)
		x_4 = tf.keras.layers.MaxPooling1D(pool_size=self.pool_size)(x_4)
		x_4 = tf.keras.layers.Flatten()(x_4)
		x_4 = tf.keras.layers.Dropout(self.dropout)(x_4)
		# concatenate all five convolution blocks
		x_all = tf.keras.layers.Concatenate()([x_0, x_1, x_2, x_3, x_4])
		x_all = tf.keras.layers.Dense(2000, activation='relu')(x_all)
		x_all = tf.keras.layers.Dense(1000, activation='relu')(x_all)
		x_all = tf.keras.layers.Dense(500, activation='relu')(x_all)
		x_all = tf.keras.layers.Dropout(self.dropout)(x_all)		
		# output layer
		outputs = tf.keras.layers.Dense(3, activation='softmax')(x_all)
		self.model = tf.keras.Model(inputs=inputs, outputs=outputs)
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
		# recall and precision for class 2 (dwarf)
		recall_2=hist['recall_2']
		val_recall_2=hist['val_recall_2']
		precision_2=hist['precision_2']
		val_precision_2=hist['val_precision_2']
		f1_2=2*(precision_2*recall_2)/(precision_2+recall_2)
		val_f1_2=2*(val_precision_2*val_recall_2)/(val_precision_2+val_recall_2)
		# test score
		test_f1_0=2*(score[2]*score[3])/(score[2]+score[3])
		test_f1_1=2*(score[4]*score[5])/(score[4]+score[5])
		test_f1_2=2*(score[6]*score[7])/(score[6]+score[7])		
		# plot values above
		plt.figure(figsize=(16,11))
		plt.plot(epochs, loss, 'b.', label='loss')
		plt.plot(epochs, val_loss, 'g.', label='validation loss')
		plt.plot(epochs, f1_0, 'bo', label='F1 score class 0')
		plt.plot(epochs, val_f1_0, 'go', label='validation F1 score class 0')
		plt.plot(epochs, f1_1, 'bs', label='F1 score class 1')
		plt.plot(epochs, val_f1_1, 'gs', label='validation F1 score class 1')
		plt.plot(epochs, f1_2, 'b^', label='F1 score class 2')
		plt.plot(epochs, val_f1_2, 'g^', label='validation F1 score class 2')
		plt.xlabel('Epoch')
		plt.ylabel('Score')
		plt.title('Loss Curve for Problem: Fringing and M-star'+ ', learning rate='+ 
			   str(learning_rate)+ ', batch size='+ str(batch_size)+ 
			   ', validation split='+ str(validation_split)+
			   ' drop out rate: '+str(self.dropout)+
			   '\n test F1 score class 0: ' + str(test_f1_0)+
			   '\n test F1 score class 1: ' + str(test_f1_1)+
			   '\n test F1 score class 2: ' + str(test_f1_2)+
			   '\n test loss: ' + str(score[0]) + 
			   ', trained on class ratios: 1/3, 1/3, 1/3'+
			   ', tested on class ratios: 0.97, 0.02, 0.01')
		plt.legend()
		plt.savefig('Documents/task_12/Losscurve_'+ 
			     str(learning_rate) + str(batch_size) + str(validation_split)+ 
			     str(epochs[-1])+
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
  

  
  
