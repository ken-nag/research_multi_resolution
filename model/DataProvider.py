import os
import sys
sys.path.append('../../../')
from lib import AudioModule
import numpy as np
from sklearn.model_selection import train_test_split

class DataProvider():
        def __init__(self, data_folder_path=os.path.abspath('./../data/DSD100_DSampled/Sources'), fs=16000, valid_rate=0.2):
                self.data_folder_path = data_folder_path
                self.fs = fs
                self.test_data_folder_path = self.data_folder_path + '/Test'
                self.train_data_folder_path = self.data_folder_path + '/Dev'
                self.test_track_list = os.listdir(self.test_data_folder_path)
                self.train_track_list = os.listdir(self.train_data_folder_path)
                self.valid_rate = valid_rate
                self.train_all_bass_list   = []
                self.train_all_drums_list  = []
                self.train_all_other_list  = []
                self.train_all_vocals_list = []
                self.test_all_bass_list   = []
                self.test_all_drums_list  = []
                self.test_all_other_list  = []
                self.test_all_vocals_list = []
                self.all_data_num = None

        def load_all_train_data(self):
                for track_name in self.train_track_list:
                        track_folder_path = self.train_data_folder_path + '/' + track_name
                        bass_data, _fs   = AudioModule.read(track_folder_path + '/bass.wav', self.fs)
                        drums_data, _fs  = AudioModule.read(track_folder_path + '/drums.wav', self.fs)
                        other_data, _fs  = AudioModule.read(track_folder_path + '/other.wav', self.fs)
                        vocals_data, _fs = AudioModule.read(track_folder_path + '/vocals.wav', self.fs)
                        self.train_all_bass_list.append(bass_data)
                        self.train_all_drums_list.append(drums_data)
                        self.train_all_other_list.append(other_data)
                        self.train_all_vocals_list.append(vocals_data)

                self.all_data_num = len(self.train_all_vocals_list)
                return self.train_all_bass_list, self.train_all_drums_list, self.train_all_other_list, self.train_all_vocals_list
            
        def load_all_test_data(self):
            for track_name in self.test_track_list:
                        track_folder_path = self.test_data_folder_path + '/' + track_name
                        bass_data, _fs   = AudioModule.read(track_folder_path + '/bass.wav', self.fs)
                        drums_data, _fs  = AudioModule.read(track_folder_path + '/drums.wav', self.fs)
                        other_data, _fs  = AudioModule.read(track_folder_path + '/other.wav', self.fs)
                        vocals_data, _fs = AudioModule.read(track_folder_path + '/vocals.wav', self.fs)
                        self.test_all_bass_list.append(bass_data)
                        self.test_all_drums_list.append(drums_data)
                        self.test_all_other_list.append(other_data)
                        self.test_all_vocals_list.append(vocals_data)
            return self.test_all_bass_list, self.test_all_drums_list, self.test_all_other_list, self.test_all_vocals_list
        
        def test_data_split_and_pad(self, source, sample_len):
            source_len = len(source)
            zero_pad_num  = sample_len  - (source_len % sample_len)
            pad_source = np.append(source, np.zeros(zero_pad_num))
            iter_num = int(len(pad_source) / sample_len)
            cutted_source_array = np.zeros((iter_num, sample_len))
            for i in range(iter_num):
                cutted_source_array[i, :] = pad_source[i*sample_len:(i+1)*sample_len]
                
            return cutted_source_array


        def split_to_train_valid(self, list):
                train_list, valid_list = train_test_split(list, test_size = self.valid_rate, shuffle = False)
                return train_list, valid_list