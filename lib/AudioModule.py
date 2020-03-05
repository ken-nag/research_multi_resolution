import scipy
import scipy.signal as sg
from scipy.io import wavfile
import IPython.display
import librosa
import numpy as np
import pickle

def read(filename, fs):
      data, fs = librosa.load(filename, fs)
      return data, fs

def read_as_mono(filename, fs):
      data, fs = librosa.load(filename,  sr=fs, dtype=np.float32, mono=True)
      return data, fs

def write(filename, x, fs):
      librosa.output.write_wav(filename, x, fs)
        
def to_pickle(data, file_name):
      with open(file_name,  mode='wb') as f:
              pickle.dump(data, f)

def mixing(bass_data_array, drums_data_array, other_data_array, vocals_data_array):
      return bass_data_array + drums_data_array + other_data_array + vocals_data_array
    
def play(audio_data, fs):
      return IPython.display.Audio(audio_data, rate=fs)
        




