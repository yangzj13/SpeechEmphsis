import utils.features
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt



if __name__ == "__main__":
    y, sr = librosa.load('sogou_zsh_100001.wav', sr = 16000)
    # # test lf0
    # plt.figure()
    # f0s =  utils.features.f0(y, sr, norm = True)
    # plt.plot(f0s[400:1200], '.')
    # plt.show()
    # # # test mfcc
    # # plt.figure()
    # # mfcc = utils.features.mfcc(y, sr, norm = True)
    # # librosa.display.specshow(mfcc.T, x_axis='time', sr=sr, hop_length=80)
    # # plt.colorbar()
    # # plt.show()
    # # test energy
    # plt.figure()
    # plt.plot(utils.features.energy(y))
    # plt.show()
    # # test duration
    # plt.figure()
    # plt.plot(utils.features.duration(y), '.')
    # plt.show()
    # # test indexs
    # indexs = utils.features.indexs(y, norm = 1)
    # print(indexs.shape)
    # #plt.figure()
    # plt.plot(indexs[3])

    # plt.show()
    print(utils.features.extract_all(y).shape)