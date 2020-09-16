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
import pickle 
import copy
from scipy.interpolate import interp1d
from astropy.convolution import Gaussian1DKernel, interpolate_replace_nans, convolve
from scipy import signal
#############################################################################################
input_shape=(4200 ,1) 
min_wavelen=4400 
max_wavelen=8800
DWARF_CODE=32 # this code is a placeholder, we will use it only here
FRINGED_CODE=64 # this is the code for fringed spectra
OTHER_CODE=0 # this is the code for spectra with no comment
FILEPATH_DATA='Documents/task_11/pickle/pickle64' # filepath where the spec_data is saved
#############################################################################################	
# function to load metadata
def load_metadata():
  with fits.open('Downloads/AATSpecAllv27.fits') as metadata:
	  GAMA_SPEC_ID_LIST=metadata[1].data['SPECID'][:] # GAMA spectrum ID
		COMMENTS_FLAG_LIST=metadata[1].data['COMMENTS_FLAG'][:] # Flags
	return GAMA_SPEC_ID_LIST, COMMENTS_FLAG_LIST
  
# function to take a file path and extract all the spectrum ids (file_name) from it
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
	GAMA_SPEC_ID_LIST, COMMENTS_FLAG_LIST = load_metadata()
	# training data indices
	ind_dwarf=get_ind('/data/fs12/jziegler/spec_images/64/train/m_star/', GAMA_SPEC_ID_LIST)
	ind_fringed=get_ind('/data/fs12/jziegler/spec_images/64/train/fringed/', GAMA_SPEC_ID_LIST)
	ind_other=get_ind('/data/fs12/jziegler/spec_images/64/train/other/', GAMA_SPEC_ID_LIST)
	# test data indices
	ind_dwarf_test=get_ind('/data/fs12/jziegler/spec_images/64/test/m_star/', GAMA_SPEC_ID_LIST)
	ind_fringed_test=get_ind('/data/fs12/jziegler/spec_images/64/test/fringed/',
				 GAMA_SPEC_ID_LIST)
	ind_other_test=get_ind('/data/fs12/jziegler/spec_images/64/test/other/', GAMA_SPEC_ID_LIST)
	# in case an index appears in both training and test data, remove it from training data
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
	labels_train=tf.keras.utils.to_categorical(labels_train, num_classes=3) # labels as one-hot vectors
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
	labels_test=tf.keras.utils.to_categorical(labels_test, num_classes=3) # labels as one_hot vectors
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
	def fourier(self, x_values, y_values):
		assert len(x_values) == len(y_values)
		x_values_f=np.fft.fftfreq(len(x_values), 
					   (np.max(x_values)-np.min(x_values))/len(x_values))
		mask= x_values_f>0
		y_values_f=np.fft.fft(y_values)
		y_values_f_true=2*np.abs(y_values_f/len(x_values))
		return x_values_f[mask], y_values_f_true[mask]
	def prepped_data_lowpass(self, min_wavelen, max_wavelen, input_shape):
		# this was a try but led to worse performance so it will not be used in the end
		# remove nan values
		wavelen_prep, spec_data_prep = rem_nans(self.wavelen, self.spec_data)
		# apply lowpass filter
		sos = signal.butter(1, 0.0035, 'lowpass', output='sos')	
		spec_data_prep = signal.sosfilt(sos, spec_data_prep)
		# crop to defined wavelength
		lower_ind=np.min(np.where(wavelen_prep>=min_wavelen))
		upper_ind=np.min(np.where(wavelen_prep>=max_wavelen))
		wavelen_prep=wavelen_prep[lower_ind: upper_ind]	
		wavelen_prep=wavelen_prep[lower_ind: upper_ind]	
		# crop to input_shape (cropping all spectra to same wavelength does not guarantee
		#	same vector length, because distances between two pixels are not the same for all
		#	spectra)
		assert len(spec_data_prep) >= input_shape[0]
		while len(spec_data_prep) > input_shape[0]:
			spec_data_prep=np.delete(spec_data_prep, 0)
			wavelen_prep=np.delete(wavelen_prep, 0)
		# bring all spectra to mean_value = 0 and norm to maximum of absolute values
		spec_data_prep = spec_data_prep-np.nanmean(spec_data_prep)
		spec_data_prep=spec_data_prep/np.absolute(spec_data_prep).max()
		return wavelen_prep, spec_data_prep
	def prepped_data(self, min_wavelen, max_wavelen, input_shape):
		# remove nan values
		wavelen_prep, spec_data_prep = rem_nans(self.wavelen, self.spec_data)
		# sliding window to cut off strong emission or absorption lines
		wavelen_prep, spec_data_prep=rem_emi(wavelen_prep, spec_data_prep, 101, 71, 6, False)
		wavelen_prep, spec_data_prep=rem_emi(wavelen_prep, spec_data_prep, 81, 61, 10, False)
		# apply convolution to smooth data (this led to worse performance so it will not be used in the end)
		#kernel=Gaussian1DKernel(stddev=14)
		#spec_data_prep = convolve(spec_data_prep, kernel)
		# crop to defined wavelength
		lower_ind=np.min(np.where(wavelen_prep>=min_wavelen))
		upper_ind=np.min(np.where(wavelen_prep>=max_wavelen))
		wavelen_prep=wavelen_prep[lower_ind: upper_ind]	
		wavelen_prep=wavelen_prep[lower_ind: upper_ind]	
		# crop to input_shape (cropping all spectra to same wavelength does not guarantee
		#	same vector length, because distances between two pixels are not the same for all
		#	spectra)
		assert len(spec_data_prep) >= input_shape[0]
		while len(spec_data_prep) > input_shape[0]:
			spec_data_prep=np.delete(spec_data_prep, 0)
			wavelen_prep=np.delete(wavelen_prep, 0)
		# bring all spectra to mean_value = 0 and norm to maximum of absolute values
		spec_data_prep = spec_data_prep-np.nanmean(spec_data_prep)
		spec_data_prep=spec_data_prep/np.absolute(spec_data_prep).max()
		return wavelen_prep, spec_data_prep		
	def plot(self, x_values, y_values):
		plt.plot(x_values, y_values)
		plt.title('GAMA Spectrum, ID: '+str(self.spec_id)+', comments: '+str(self.flag))
		plt.xlabel('wavelength / Å')
		plt.ylabel('intensity / arb. units')
		plt.show()

# function to remove nan values with linear interpolation		
def rem_nans(wavelen, flux):
	wavelen_no_nans=copy.copy(wavelen)
	flux_no_nans=copy.copy(flux)
	if np.isnan(flux_no_nans[0])==True:
		flux_no_nans[0]=np.nanmean(flux_no_nans)
	if np.isnan(flux_no_nans[-1])==True:
		flux_no_nans[-1]=np.nanmean(flux_no_nans)
	interp = interp1d(wavelen_no_nans[np.isnan(flux_no_nans)==False], 
			  flux_no_nans[np.isnan(flux_no_nans)==False])
	#wavelen_no_nans = wavelen_no_nans[30:len(wavelen_no_nans)-30]
	flux_no_nans = interp(wavelen_no_nans) 
	return wavelen_no_nans, flux_no_nans

# function to remove emission/absorption lines with sliding window
def rem_emi(wavelen, flux, window_size, ex_size, lim, crop):
	pad=int(window_size/2)
	ex_05=int(ex_size/2)
	for i in range(pad, len(flux)-pad):
		window=flux[i-pad:i+pad+1]
		window_ex=np.append(window[:pad-ex_05], window[pad+ex_05+1:])
		lower_limit=-lim*np.nanstd(window_ex)+np.nanmean(window_ex)
		upper_limit=lim*np.nanstd(window_ex)+np.nanmean(window_ex)
		if flux[i]<lower_limit:
			flux[i]=np.nanmean(window_ex)
		if flux[i]>upper_limit:
			flux[i]=np.nanmean(window_ex)
	if crop==True:
		flux=flux[pad:len(flux)-pad]
		wavelen=wavelen[pad:len(wavelen)-pad]
	return wavelen, flux

#############################################################################################
if __name__ == "__main__":
  spec_data_train, labels_train, spec_data_test, labels_test, train_ind, test_ind = make_data_sets(min_wavelen, max_wavelen, input_shape, FILEPATH_DATA)
  
