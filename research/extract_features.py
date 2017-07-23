from __future__ import print_function
import utils.features
import librosa
import librosa.display
import numpy as np
import sys
import os

time_stamps = 2000

if __name__ == "__main__":
	if (len(sys.argv) != 3):
		print("usage: python extract_features.py <audio_dir>, <target_name(*.npz)>")
	else:
		dataDir = sys.argv[1]
		dataName = sys.argv[2]
		fs = []
		labels = []
		files = os.listdir(dataDir + '\\' + 'wav')
		for i in range(len(files)):
			file =  files[i]
			filename = file[:file.index('.')]
			audiopath = os.path.join(dataDir, 'wav', filename+'.wav')
			monopath = os.path.join(dataDir, 'mono', filename+'.lf0')
			print("handling ", audiopath, end = '')
			##audio feature
			y, sr = librosa.load(audiopath, sr = 16000)
			feature = utils.features.extract_all(y).T
			if feature.shape[0] > time_stamps:
				fs.append(feature[:time_stamps, :])
			else:
				fs.append(np.vstack((feature, np.zeros((time_stamps - feature.shape[0], feature.shape[1])))))

			##mono label
			mono = np.loadtxt(monopath, usecols=(0,1,3), dtype = np.int)
			em = mono[:, 2]
			start = mono[:, 0]
			end = mono[:, 1]
			label = np.zeros((time_stamps, 1))
			for w in range(mono.shape[0]):
				for k in range(start[w], end[w]):
					label[k][0] = em[w]
			labels.append(label)

			print("   Done")

		fs = np.array(fs)
		labels = np.array(labels)
		print(fs.shape, labels.shape)
		print("data shape =", fs.shape[:2])
		np.savez(dataName, X = fs, Y = labels)
	

