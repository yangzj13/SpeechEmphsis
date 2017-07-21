from __future__ import print_function
import utils.features
import librosa
import librosa.display
import numpy as np
import sys
import os

if __name__ == "__main__":
	if (len(sys.argv) != 3):
		print("usage: python extract_features.py <audio_dir>, <target_name(*.npy)>")
	else:
		dataDir = sys.argv[1]
		dataName = sys.argv[2]
		fs = []
		for file in os.listdir(dataDir):

			filepath = os.path.join(dataDir, file)
			print("handling ", filepath)
			y, sr = librosa.load(filepath, sr = 16000)
			fs.append(utils.features.extract_all(y).T)

		fs = np.array(fs)
		print("data shape =", fs.shape)
		np.save(dataName, fs)
	

