import numpy as np
import librosa
import pyaudio

class Audio:
    
    def __init__(self):
        pass
    
    def read_file(self, path, sample_rate = 44100):
        
        """ 
        Read a file from the file path and returns a numpy array of song data.

        Parameters
        ----------
        path : global path to file
        sample_rate : int
        
        Returns
        -------
        numpy.ndarray

        """

        song, song_sample_rate = librosa.load(path, sr=sample_rate, mono=True)
        return song, song_sample_rate

    def read_mic(self, seconds=5):
        
        """ 
        Read 'seconds' seconds of audio from the mic and returns a numpy array.

        Parameters
        ----------
        secconds : seconds of audio to record
        
        Returns
        -------
        Array of audio data

        """
        
        CHUNK = 1024
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 44100 # read at twice the input rate
        
        p = pyaudio.PyAudio()
        
        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)
        
        npaudio = np.fromstring(stream.read(RATE*seconds), dtype=np.int16)

        # stop recording, close
        stream.stop_stream()
        stream.close()
        
        p.terminate()
        
        return npaudio
    
    def play_audio(self, soundArray):
        
        """ 
        Play audio from a numpy array

        Parameters
        ----------
        soundArray : numpy array

        """

        CHUNK = 1024
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 44100 # read at twice the input rate
        
        p = pyaudio.PyAudio()

        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        output=True,
                        frames_per_buffer=CHUNK)

        # repeat the data
        stream.write(soundArray, num_frames=None, exception_on_underflow=False)

        # stop and close
        stream.stop_stream()
        stream.close()
        
        p.terminate()
        pass
    
