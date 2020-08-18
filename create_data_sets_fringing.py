# import libraries
import astropy # library with useful tools often needed in astrophysics
from astropy.io import fits # fits is the data format used here (most common in astrophysics)
import matplotlib.pyplot as plt # makes plots
import numpy as np # library with useful commands for arrays of numbers
import tensorflow as tf # useful commands for machine learning
import pandas as pd # library to manage data tables
import glob # to create list of file names
from scipy.optimize import curve_fit # library for curve fits
import math
import os
import pickle 
from scipy.interpolate import interp1d
from astropy.convolution import Gaussian1DKernel, interpolate_replace_nans, convolve
#############################################################################################
input_shape=(4200 ,1) 
min_wavelen=4400 
max_wavelen=8800
DWARF_CODE=32 # this code is a placeholder, we will use it only here
FRINGED_CODE=64 # this is the code for fringed spectra
OTHER_CODE=0 # this is the code for spectra with no comment
FILEPATH_DATA='Documents/task_10/pickle/pickle64_sharp_set_smooth_3' # filepath where the spec_data is saved
#############################################################################################	
# function to load metadata
def load_metadata():
  with fits.open('Downloads/AATSpecAllv27.fits') as metadata:
	  GAMA_SPEC_ID_LIST=metadata[1].data['SPECID'][:] # GAMA spectrum ID
		COMMENTS_FLAG_LIST=metadata[1].data['COMMENTS_FLAG'][:] # Flags
	return GAMA_SPEC_ID_LIST, COMMENTS_FLAG_LIST
  
# function to take a file path('.../') and extract all the spectrum ids (file_name) from it
# spectra are sorted into different folders regarding their class (other/fringed/m-star)
# the file_name will be compared to gama_spec_id and the respective indices are returned 
def get_ind(filepath, GAMA_SPEC_ID_LIST):
	assert type(filepath) is str
	file_name=glob.glob(filepath+'*.png')
	file_ind=np.zeros(len(file_name))
	for i in range(len(file_name)):
		file_name[i]=os.path.splitext(file_name[i])[0][len(filepath):]
		file_ind[i]=np.where(GAMA_SPEC_ID_LIST==file_name[i])[0][0]
	file_ind=file_ind.astype(int)
	np.random.shuffle(file_ind) 
	return file_ind
  
# function to create training and test data sets from the indices
def make_data_sets(min_wavelen, max_wavelen, input_shape, filepath):
	assert type(filepath) is str
	GAMA_SPEC_ID_LIST, COMMENTS_FLAG_LIST = load_metadata()
	# training data indices
	ind_dwarf=get_ind('/data/fs12/jziegler/spec_images/64/train/m_star/', GAMA_SPEC_ID_LIST)
	ind_fringed=get_ind('/data/fs12/jziegler/spec_images/64/train/fringed/', GAMA_SPEC_ID_LIST)
	ind_other=get_ind('/data/fs12/jziegler/spec_images/64/train/other/', GAMA_SPEC_ID_LIST)
	ind_other=np.append(ind_other, get_ind(
			'/data/fs12/jziegler/spec_images/64/train/other?/', GAMA_SPEC_ID_LIST))
	# test data indices
	ind_dwarf_test=get_ind('/data/fs12/jziegler/spec_images/64/test/m_star/', GAMA_SPEC_ID_LIST)
	ind_fringed_test=get_ind('/data/fs12/jziegler/spec_images/64/test/fringed/',
				 GAMA_SPEC_ID_LIST)
	ind_other_test=get_ind('/data/fs12/jziegler/spec_images/64/test/other/', GAMA_SPEC_ID_LIST)
	ind_other_test=np.append(ind_other_test, get_ind(
				'/data/fs12/jziegler/spec_images/64/test/other?/', GAMA_SPEC_ID_LIST))
	# in case an index appears in training and test data, remove it from training data
	ind_dwarf=np.setdiff1d(ind_dwarf, ind_dwarf_test)
	ind_fringed=np.setdiff1d(ind_fringed, ind_fringed_test)
	ind_other=np.setdiff1d(ind_other, ind_other_test)
	# make sure now no index appears in different arrays of indices twice
	LIST=[ind_dwarf, ind_fringed, ind_other, ind_dwarf_test, ind_fringed_test, ind_other_test]
	for i in range(len(LIST)):
		for j in range(len(LIST)):
			if i!=j:
				assert (LIST[i][:len(LIST[j])]==LIST[j][:len(LIST[i])]).any()==False
	# correct the comments in 'comments_fag' with the corresponding comment code
	COMMENTS_FLAG_LIST[ind_dwarf]=DWARF_CODE
	COMMENTS_FLAG_LIST[ind_fringed]=FRINGED_CODE
	COMMENTS_FLAG_LIST[ind_other]=OTHER_CODE
	COMMENTS_FLAG_LIST[ind_dwarf_test]=DWARF_CODE
	COMMENTS_FLAG_LIST[ind_fringed_test]=FRINGED_CODE
	COMMENTS_FLAG_LIST[ind_other_test]=OTHER_CODE
	# create training set (balanced)
	MAX_LENGTH=400
	for i in range(int(len(LIST)/2)):
		assert MAX_LENGTH <= len(LIST[i]) 
	train_ind=np.append(ind_dwarf[:MAX_LENGTH], ind_fringed[:MAX_LENGTH])
	train_ind=np.append(train_ind, ind_other[:MAX_LENGTH])
	np.random.shuffle(train_ind)
	labels_train=np.zeros((len(train_ind), ))
	spec_data_train=np.zeros((len(train_ind), input_shape[0]))
	for i in range(len(train_ind)):
		SPEC=spectra(GAMA_SPEC_ID_LIST, COMMENTS_FLAG_LIST, train_ind[i])
		labels_train[i]=SPEC.flag
		wavelen, spec_data_train[i]=SPEC.prepped_data(min_wavelen, max_wavelen, input_shape)
	labels_train=labels_train.astype(int)
	labels_train[labels_train==OTHER_CODE]=0
	labels_train[labels_train==FRINGED_CODE]=1
	labels_train[labels_train==DWARF_CODE]=2	
	labels_train=tf.keras.utils.to_categorical(labels_train, num_classes=3)
	# create test set (unbalanced/random distribution)
	test_ind=np.append(ind_dwarf_test, ind_fringed_test)
	test_ind=np.append(test_ind, ind_other_test)
	np.random.shuffle(test_ind)
	labels_test=np.zeros((len(test_ind), ))
	spec_data_test=np.zeros((len(test_ind), input_shape[0]))
	for i in range(len(test_ind)):
		SPEC=spectra(GAMA_SPEC_ID_LIST, COMMENTS_FLAG_LIST, test_ind[i])
		labels_test[i]=SPEC.flag
		wavelen, spec_data_test[i]=SPEC.prepped_data(min_wavelen, max_wavelen, input_shape)
	labels_test=labels_test.astype(int)
	labels_test[labels_test==OTHER_CODE]=0
	labels_test[labels_test==FRINGED_CODE]=1
	labels_test[labels_test==DWARF_CODE]=2
	labels_test=tf.keras.utils.to_categorical(labels_test, num_classes=3)
	# save data sets
	with open(filepath, 'wb') as pkl_out:
		pickle.dump(spec_data_train, pkl_out, pickle.HIGHEST_PROTOCOL)
		pickle.dump(labels_train, pkl_out, pickle.HIGHEST_PROTOCOL)
		pickle.dump(train_ind, pkl_out, pickle.HIGHEST_PROTOCOL)
		pickle.dump(spec_data_test, pkl_out, pickle.HIGHEST_PROTOCOL)
		pickle.dump(labels_test, pkl_out, pickle.HIGHEST_PROTOCOL)
		pickle.dump(test_ind, pkl_out, pickle.HIGHEST_PROTOCOL)
	return spec_data_train, labels_train, spec_data_test, labels_test, train_ind, test_ind
#############################################################################################
# class to get relevant info for one spectrum
class spectra:
	def __init__(self, GAMA_SPEC_ID_LIST, COMMENTS_FLAG_LIST, i):
		assert len(GAMA_SPEC_ID_LIST)==len(COMMENTS_FLAG_LIST)		
		assert i <= len(GAMA_SPEC_ID_LIST)
		self.spec_id=GAMA_SPEC_ID_LIST[i]
		self.flag=COMMENTS_FLAG_LIST[i]
		with fits.open('/data/fs12/gama/spectra/gama/reduced_27/1d/'+ 
				self.spec_id +'.fit') as all_data:
			self.spec_data=all_data[0].data[0]
			self.error=all_data[0].data[1]
			self.header=all_data[0].header
		self.wmin=self.header['WMIN']
		self.wmax=self.header['WMAX']
		self.wavelen=np.arange(start=self.wmin, stop=self.wmax, 
				       step=((self.wmax-self.wmin)/len(self.spec_data)))
	def spec_id(self):
		return self.spec_id
	def flag(self):
		return self.flag
	def spec_data(self):
		return self.spec_data
	def wavelen(self):
		return self.wavelen
	def fourier(self, x_values, y_values):
		assert len(x_values) == len(y_values)
		x_values_f=np.fft.fftfreq(len(x_values), 
					   (np.max(x_values)-np.min(x_values))/len(x_values))
		mask= x_values_f>0
		y_values_f=np.fft.fft(y_values)
		y_values_f_true=2*np.abs(y_values_f/len(x_values))
		return x_values_f[mask], y_values_f_true[mask]
	def prepped_data(self, min_wavelen, max_wavelen, input_shape):
		assert len(self.wavelen)==len(self.spec_data)
		lower_ind=np.min(np.where(self.wavelen>=min_wavelen))
		upper_ind=np.min(np.where(self.wavelen>=max_wavelen))
		wavelen_cropped=self.wavelen[lower_ind: upper_ind]
		spec_data_cropped=self.spec_data[lower_ind: upper_ind]
		#spec_data_cropped[np.isnan(spec_data_cropped)]=0
		interp = interp1d(self.wavelen[np.isnan(self.spec_data)==False], 
				  self.spec_data[np.isnan(self.spec_data)==False])
		spec_data_cropped = interp(wavelen_cropped) # interpolate to remove nan values
		assert len(spec_data_cropped) >= input_shape[0]
		while len(spec_data_cropped) > input_shape[0]:
			spec_data_cropped=np.delete(spec_data_cropped, 0)
			wavelen_cropped=np.delete(wavelen_cropped, 0)
		lower_limit=-4*spec_data_cropped.std()+spec_data_cropped.mean()
		upper_limit=4*spec_data_cropped.std()+spec_data_cropped.mean()
		spec_data_cropped[spec_data_cropped<lower_limit]=lower_limit
		spec_data_cropped[spec_data_cropped>upper_limit]=upper_limit
		spec_data_cropped=spec_data_cropped/np.absolute(spec_data_cropped).max()
		# apply convolution to smooth data
		kernel=Gaussian1DKernel(stddev=3) #9
		spec_data_cropped = convolve(spec_data_cropped, kernel)
		return wavelen_cropped, spec_data_cropped
	def plot(self, x_values, y_values):
		self.y_values=y_values
		self.x_values=x_values
		assert len(self.x_values) == len(self.y_values)
		plt.plot(self.x_values, self.y_values)
		plt.title('GAMA Spectrum, ID: '+str(self.spec_id)+', comments: '+str(self.flag))
		plt.xlabel('wavelength / Å')
		plt.ylabel('intensity / arb. units')
		plt.show()
#############################################################################################
if __name__ == "__main__":
  spec_data_train, labels_train, spec_data_test, labels_test, train_ind, test_ind = make_data_sets(min_wavelen, max_wavelen, input_shape, FILEPATH_DATA)
  
