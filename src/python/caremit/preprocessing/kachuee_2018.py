from scipy.signal import decimate
from scipy.signal import find_peaks
import numpy as np

samples_for_10seconds_of_125hz = 125*10

def minmax_to_01(x, **kwargs):
    _min, _max = x.min(**kwargs, keepdims=True), x.max(**kwargs, keepdims=True)
    return (x - _min) / (_max - _min)

def make_odd(x):
    return x if x % 2 != 0 else x + 1

def preprocess_kachuee_2018(single_signal, fs, fixed_segment_len = 187, heartbeat_position="left"):
    """ Applies the preprocessing of the paper Kachuee et.al. 2018 https://arxiv.org/pdf/1805.00794.pdf

    Finds heartbeats, cuts out a segment of the signal around each heartbeat,
    and normalizes the heartbeat signal by downsampling, minmax, and padding.

    Args:
        single_signal (numpy.array): signal data as float values in a 1-dimensional numpy array
        fs (int): sampling frequency of `single_signal`
        fixed_segment_len (int, optional): segments will have this specified length
        heart_beat_position (string): optional, either "left" (default) or "center".

    Returns:
        segments, start_end:
            two 2D arrays, the first, `segments`, contains the segments on each row (one segment for each heartbeat)
            the second, `start_end`, contains two columns start and end which specify the indices of the segment within the original data.
    """
    heartbeat_position = heartbeat_position.lower()
    sample_factor = round(fs/125)
    signal_downsampled = decimate(single_signal, sample_factor)
    
    _muliple_padded = len(signal_downsampled) + (samples_for_10seconds_of_125hz - len(signal_downsampled) % samples_for_10seconds_of_125hz)
    signal_downsampled_padded = np.zeros(_muliple_padded, dtype = signal_downsampled.dtype)
    signal_downsampled_padded[:len(signal_downsampled)] = signal_downsampled

    linear = signal_downsampled_padded
    windowed = linear.reshape((-1, samples_for_10seconds_of_125hz))
    windowed_normalized = minmax_to_01(windowed, axis=1)

    peaks_windowed_indexes = []
    for i, window in enumerate(windowed_normalized):
        peaks, properties = find_peaks(window, height=0.9)
        peaks_windowed_indexes += [peaks + i*len(window)]

    _segmented_normalized = []
    _starts = []
    _ends = []
    for peaks_indexes in peaks_windowed_indexes:
        if len(peaks_indexes) <= 1:
            continue # TODO improve over this, as it may indeed be important to look at all data

        T = np.median(np.diff(peaks_indexes))
        window_size = round(T*1.2)
        if heartbeat_position == "center":
            window_size = make_odd(window_size)
            start = np.maximum(0, peaks_indexes - window_size//2)
            end = np.minimum(len(linear), peaks_indexes + window_size//2 + 1)
        elif heartbeat_position == "left":
            start = peaks_indexes
            end = np.minimum(len(linear), peaks_indexes + window_size)
        else:
            raise ValueError(f"Got unsupported heartbeat_position: '{heartbeat_position}'")

        # for finding the segments in the original data, we provide start and end indices resampled to the original timeseries
        _starts += list(start * sample_factor)
        _ends += list(end * sample_factor)

        for s, e in zip(start, end):
            _segmented_normalized += [minmax_to_01(linear[s:e][:fixed_segment_len])]
        
    segmented_normalized = np.zeros((len(_segmented_normalized), fixed_segment_len), dtype=linear.dtype)
    for i, row in enumerate(_segmented_normalized):
        segmented_normalized[i, :len(row)] = row
    
    return segmented_normalized, np.c_[_starts, _ends]