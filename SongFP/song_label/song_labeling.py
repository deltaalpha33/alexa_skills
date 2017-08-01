import os
import matplotlib.mlab as mlab
import numpy as np
from collections import Counter, defaultdict
from fourier import discrete_transform
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion 
from scipy.ndimage.morphology import iterate_structure
import pickle

### db has list of ( (fp part), "name" )
class FingerPrinter():
    def __init__(self):
        ### Database is of the form (f_i , f_j , dt): [ (songname, t_f_i), ... ]
        self.Database = defaultdict(list)
        self.names = []
    def add_song(self,peaks,name,lookahead = 25):
        """
        Add the fingerprints of a given set of peaks to the database with a name.
        lookahead is how many peaks ahead it should match.
        """
        if name not in self.names:
            self.names.append(name)
            ind = self.names.index(name)
            freqs = peaks[0]
            times = peaks[1]
            for i in range(0,len(freqs)-1-lookahead):
                for j in range(i+1,i+lookahead+1):
                    self.Database[(freqs[i],freqs[j],times[j]-times[i])].append( (ind, times[i]) )
    def get_peaks(self,sample):
        """
        Get the peaks of a sampled song.
        """
        S, freqs, times = mlab.specgram(sample, NFFT=4096, Fs=44100,
                                                            window=mlab.window_hanning,
                                                            noverlap=(4096 // 2))

        ys, xs = np.histogram(S.flatten(), bins=len(freqs)//2, normed=True)
        dx = xs[-1] - xs[-2]
        cdf = np.cumsum(ys)*dx  # this gives you the cumulative distribution of amplitudes
        cutoff = xs[np.searchsorted(cdf, 0.77)]

        foreground =  (S >= cutoff)

        pks = self.local_peaks(S) & foreground
        return np.where(pks)
    def local_peaks(self,data):
        """ Find local peaks in a 2D array of data.

        Parameters
        ----------
        data : numpy.ndarray

        Returns
        -------
        Binary indicator, of the same shape as `data`. The value of
        True indicates a local peak. """
        struct = generate_binary_structure(2, 1)
        neighborhood = iterate_structure(struct, 20)  # this incorporates roughly 20 nearest neighbors
        acceptable_values = data != 0
        peaks = ((maximum_filter(data, footprint=neighborhood)) == data)
        acceptable_peaks = np.logical_and(peaks, acceptable_values)
        return acceptable_peaks
    def get_finger_print(self,peaks):
        """
        Get the fingerprints of the peaks of a song.
        """
        freqs = peaks[0]
        times = peaks[1]
        lst = []
        for i in range(0,len(freqs)-1):
            for j in range(i+1,len(freqs)):
                lst.append( (freqs[i],freqs[j],times[j]-times[i]) )
        return lst
    def match_song(self,fingerprint):
        """
        Return all matches of the fingerprints provided in the database, ranked by number of hits.
        """
        c = Counter()
        for prnt in fingerprint:
            c.update(self.Database[prnt])
        return c.most_common()
    def best(self,most_common,k=10):
        """
        Return best name from top k results of match_song
        """
        c = Counter(most_common[:k][0][0])
        return self.names[c.most_common(1)[0][0]]
    def loadDBsongs(self,dirt,audio):
        """
        Load a database of songs using Audio class audio
        """

        rootDir = dirt
        print(rootDir)
        
        fileSet = set()



        for dir_, _, files in os.walk(rootDir):
            for fileName in files:
                relDir = os.path.relpath(dir_, rootDir)
                relFile = os.path.join(rootDir, fileName)
                if not fileName.startswith('.'):
                    print(fileName)
                    fileSet.add( (relFile, fileName) )
        for file in fileSet:
            
            name = file[1].replace('_',' ')
            
            print(name)
            
            samples = audio.read_file(file[0])[0]
            print(samples)
            self.add_song(self.get_peaks( samples ), name)
    def saveDB(self,dirt,dirNames):
        """
        Save the current list of names and fingerprint database to directories dirt and dirNames
        """
        pickle.dump( self.Database, open( dirt, "wb"))
        pickle.dump( self.names, open( dirNames, "wb"))
    def loadDB(self,dirt,dirNames):
        self.Database = pickle.load( open (dirt,"rb"))
        self.names = pickle.load( open (dirNames,"rb"))