import librosa
import numpy as np

def smooth(x,window_len=5,window='hamming'):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError, "smooth only accepts 1 dimension arrays."

    if x.size < window_len:
        raise ValueError, "Input vector needs to be bigger than window size."


    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"


    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    return y

def lf0(y, sr = 16000, hop_length = 80, n_fft = 512, threshold = 0.9, fmin = 100, fmax = 600):
    """
    calculate log f0 of audio file
    """
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr,
                                           n_fft=n_fft, hop_length=hop_length,
                                           threshold=threshold, fmin=fmin, fmax=fmax)
    pitch = np.zeros(pitches.shape[1])
    for t in range(pitches.shape[1]):
        tmp = pitches[:, t]
        if (len(np.unique(tmp)) == 1):
            pitch[t] = 1.0
        else:
            pitch[t] = tmp[np.nonzero(tmp)[0][0]]

    mask = np.zeros(len(pitch))
    for t in range(len(mask) - 1):
        if pitch[t + 1] / pitch[t] > 100 or pitch[t] / pitch[t + 1] > 100:
            mask[t] = 1
    pitch[mask == 1] = 1
    pitch = np.log(smooth(pitch))

    return pitch
