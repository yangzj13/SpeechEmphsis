from __future__ import print_function
import keras
from keras.models import Sequential
import keras.utils.np_utils as kutils
from keras.layers import Dense, Dropout, Flatten, Activation, TimeDistributed, Bidirectional
from keras.layers import Conv2D, MaxPooling2D, LSTM, Masking, Embedding, BatchNormalization
import os, sys
import numpy as np
import keras.backend as K
import h5py
import librosa

def ext_feature(y, )

def emphasis(feature, mono, threshold = 0.3):
	"""
	***input shape***
	feature 	: [time_stamps, n_features]
	mono    	: [n_words, 3](<string>word text, <float>start_time , <float>end_time )
	threshold 	: 0-1 float
	***output shape**
	emphasis    : [n_words, 2](<string>word text, <float>propebility of emphasis)
	"""

	model = model_from_json(open('emphasis_architecture.json').read())    
	model.load_weights('emphasis_weights.h5')    

	pred = model.predict(feature[np.newaxis, :])[0,:,1]

	mono[:, 1] = librosa.core.time_to_frames(mono[:, 1], sr = 16000, hop_length = 80, n_fft = 512)
	mono[:, 2] = librosa.core.time_to_frames(mono[:, 2], sr = 16000, hop_length = 80, n_fft = 512)

	em = []
	for i in range(len(mono)):
		start_time = mono[i][1]
		end_time = mono[i][2]
		p = np.mean(pred[start_time:end_time])
		em.append([mono[i][0], p])

	return em
