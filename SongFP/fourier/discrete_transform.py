#import numpy
import numpy as np
import numpy.ma as ma
import matplotlib.mlab as mlab

#import scipy functions for peak finding

from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion 
from scipy.ndimage.morphology import iterate_structure



def spectrogram(samples):
    """create a spectrogram from samples

    Parameters
    ----------
    samples: the data to build the spectrogram from


    Returns
    -------
    S: the spectrogram as a numpy array Rows are times, columns are frequencies
    frecs: the frequencies
    times: the times"""
    S, freqs, times = mlab.specgram(samples, NFFT=4096, Fs=44100,
                                                    window=mlab.window_hanning,
                                                    noverlap=(4096 // 2))
    return S, freqs, times

def filterlowamplitudes(C, threshold):
    """ Read

        Parameters
        ----------
        C: 2D numpy array of amplitudes from FFT
            Time on x, Frequency on y
        threshold: Threshold to determine high or low
        Returns
        -------
        2D boolean mask, True is high, False is low """
    m = np.copy(C)
    u = m < threshold
    ge = m > threshold
    m[u] = False
    m[ge] = True
    return m

def local_peaks(data):
    """ Find local peaks in a 2D array of data.

    Parameters
    ----------
    data : numpy.ndarray
    
    Returns
    -------
    Binary indicator, of the same shape as `data`. The value of
    True indicates a local peak. 
    """
    struct = generate_binary_structure(2, 1)
    neighborhood = iterate_structure(struct, 20) 
    ##acceptable_values = data > 1
    peaks = data == (maximum_filter(data, footprint=neighborhood))
    ##acceptable_peaks = np.logical_and(peaks, acceptable_values)
    ##return acceptable_peaks
    return peaks

def pair_frequencies(data, peaks , look_ahead = 15):
    """ Calculate Differences between  to sets of peaks represented  in 2D arrays of data.

    Parameters
    ----------
    data : numpy.ndarray
    peaks : numpy.ndarray 
        mask of peaks
    look_ahead : int
        how many other points of data to consider

    Returns
    -------
    Binary indicator, of the same shape as `data`. The value of
    True indicates a local peak. 
    """

    masked_data = ma.masked_array(data,  mask=peaks)

    time_array = np.hsplit(x, x.shape[0]) #list of numpy arrays
    
    analyzed_freq = list()
    for current_time in range(len(time_array) - look_ahead): #dont compare last element(s)
        current_frequencies = time_array[current_time]
        for frequency1 in current_frequencies:
            
            for compare_time in range(look_ahead):
                compare_frequencies = time_array[compare_time]
                for frequency2 in compare_frequencies:
                    delta_t = compare_time - current_time
                    analyzed_freq.append((frequency1, frequency2, delta_t))
    return analyzed_freq













