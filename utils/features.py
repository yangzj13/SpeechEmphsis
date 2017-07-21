import librosa
import numpy as np

def normalize_f0(a, threshold = 1e-9):
    """
    steps:
    1. remove items <  threshold
    2. calc normlization result
    3. ignore those items < threshold by make them still be 0
    
    """
    mask = a > threshold
    tmp = a[mask]
    a[a < threshold] = np.mean(tmp)
    a = (a - np.mean(tmp)) / np.std(tmp)
    return a

def f0(y, sr = 16000, hop_length = 80, n_fft = 512, threshold = 0.2, fmin = 100, fmax = 600, norm = True):
    """
    return shape:
    (n_frames, )
    """
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr,
                                           n_fft=n_fft, hop_length=hop_length,
                                           threshold=threshold, fmin=fmin, fmax=fmax)
    pitch = np.zeros(pitches.shape[1])
    for t in range(pitches.shape[1]):
        tmp = pitches[:, t]
        if (len(np.unique(tmp)) == 1):
            pitch[t] = 0
        else:
            pitch[t] = tmp[np.nonzero(tmp)[0][0]]

    mask = np.zeros(len(pitch))
    for t in range(0, len(mask) - 1, 1):
        ave = np.average(pitch[t:t+5])
        for k in range(5):
            if t + k < len(pitch) and np.abs(pitch[t + k] - ave) > ave/4:
                mask[t + k] = 1
    pitch[mask == 1] = 0

    if norm :
        pitch = normalize_f0(pitch)
        #pitch = librosa.util.normalize(pitch)

    return pitch

def mfcc(y, sr = 16000, hop_length = 80, n_fft = 512, n_mfcc = 13, rm_first = True, norm = True):
    """
    return shape:
    (n_features = n_mfcc - 1 = 12, n_frames)
    """
    mfccs = librosa.feature.mfcc(y, sr, hop_length=hop_length, n_fft = n_fft, n_mfcc =  n_mfcc)
    if norm:
#        mfccs = librosa.util.normalize(mfccs, axis = 0)
        for fea in range(n_mfcc):
            mfccs[fea] = librosa.util.normalize(mfccs[fea])
    
    if rm_first : 
        mfccs = mfccs[1:]

    return mfccs

def energy(y, sr = 16000, frame_length = 400, hop_length = 80, n_fft = 512, norm = True):
    """
    return shape:
    (n_features = 1, n_frames)
    """
    energy = librosa.feature.rmse(y, frame_length = frame_length, hop_length = hop_length, n_fft = n_fft)
    energy = librosa.util.normalize(energy, axis = 1)
    #energy = normalize_f0(energy, 1e-2)
    #energy = (energy - np.mean(energy)) / np.std(energy)
    return energy

def duration(y, sr = 16000, frame_length = 400, hop_length = 80, n_fft = 512, norm = True):
    """
    return shape:
    (n_frames,)
    """
    _f0 = f0(y, sr, hop_length = hop_length, n_fft = n_fft, norm = False)
    _length = _f0.shape[0]
    _duration = np.zeros((_length,))
    _duration[_f0 > 1e-9] = 1

    for t in range(1, _length, 1):
        _duration[t] = _duration[t] * (_duration[t - 1] + _duration[t])

    for t in range(_length - 2, 0, -1):
        if _duration[t + 1] > 0:
            _duration[t] = _duration[t + 1] * (_duration[t] > 0)
    if norm:
        _duration = normalize_f0(_duration, 5)

    return _duration

def indexs(y, sr = 16000, frame_length = 400, hop_length = 80, n_fft = 512, norm = True):
    """
    return shape:
    (n_features = 4, n_frames)
    """
    _duration = duration(y, sr = sr, frame_length = frame_length, hop_length = hop_length, norm = False)
    _duration[_duration < 5] = 0
    n_frames = _duration.shape[0]
    indexs_sent_head = np.array(range(n_frames))
    indexs_sent_tail = n_frames - indexs_sent_head
    indexs_phom_head = np.copy(_duration)
    for t in range(n_frames - 2, 0, -1):
        if indexs_phom_head[t + 1] > 0:
            indexs_phom_head[t] = (indexs_phom_head[t + 1] - 1) * (indexs_phom_head[t] > 0)
    indexs_phom_tail = _duration - indexs_phom_head

    if norm:
        indexs_sent_head = normalize_f0(indexs_sent_head)
        indexs_sent_tail = normalize_f0(indexs_sent_tail)
        indexs_phom_head = normalize_f0(indexs_phom_head)
        indexs_phom_tail = normalize_f0(indexs_phom_tail)

    return np.vstack((indexs_sent_head, indexs_sent_tail, indexs_phom_head, indexs_phom_tail))

def extract_all(y, sr = 16000, frame_length = 400, hop_length = 80, n_fft = 512, norm = True):
    """
    return shape:
    (n_features = 19, n_frames)
    features include (f0, mfcc, energy, duration, indexs)
    """
    _f0 = f0(y, sr = sr, hop_length = hop_length, n_fft = n_fft, norm = norm)
    _mfcc = mfcc(y, sr = sr, hop_length = hop_length, n_fft = n_fft, norm = norm)
    _energy = energy(y, sr = sr, frame_length = frame_length, hop_length = hop_length, n_fft = n_fft, norm = norm)
    _duration = duration(y, sr = sr, frame_length = frame_length, hop_length = hop_length, n_fft = n_fft, norm = norm)
    _indexs = indexs(y, sr = sr, frame_length = frame_length, hop_length = hop_length, n_fft = n_fft, norm = norm)
    
    return np.vstack((_f0[np.newaxis, :], _mfcc, _energy, _duration[np.newaxis, :], _indexs))    
