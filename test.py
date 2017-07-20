import utils.features
import librosa
import matplotlib.pyplot as plt



if __name__ == "__main__":
    y, sr = librosa.load('sogou_zsh_100001.wav', sr = 16000)
    lf0s =  utils.features.lf0(y, sr)
    plt.plot(lf0s)
    plt.show()
