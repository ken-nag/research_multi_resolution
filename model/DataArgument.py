#random mixing of instruments from diffeernt song
#random scaling with uniform amplitudes [0.25, 1.25]
#random chunking into sequences for each instrument, and,
#shuffle

import numpy as np
import random

class DataArgument():
    def __init__(self, fs, sec, data_num):
        self.fs = fs
        self.sec = sec
        self.data_num = data_num

    def __call__(self, sources_list):
        list_len = len(sources_list)
        iter_num = int(self.data_num / list_len)
        arg_list = np.zeros((self.data_num, self.fs*self.sec))
        for i in range(iter_num):
            shuffled = self.shuffle_list(sources_list)
            cutted = self.random_cutting(shuffled)
            scaled = self.random_amplitude_scaling(cutted)
            arg_list[i*list_len:(i+1)*list_len, :] = scaled[:, :]
        return arg_list

    def shuffle_list(self, sources_list):
        return np.random.permutation(sources_list)

    def random_amplitude_scaling(self, sources_list):
        list_len = len(sources_list)
        random_coeff = np.expand_dims(np.random.uniform(0.25, 1.25, list_len), axis=-1)
        scaled_sources_list = np.multiply(sources_list, random_coeff)
        return scaled_sources_list

    def random_cutting(self, sources_list):
        list_len = len(sources_list)
        cut_sources_array = np.zeros((list_len, self.fs*self.sec))
        for i, source_data in enumerate(sources_list):
            source_len = len(source_data)
            source_data = source_data.reshape(1, -1)
            offset = random.randrange(source_len - self.fs*self.sec)
            cut_sources_array[i, :] = source_data[:, offset:offset+self.fs*self.sec]
        return cut_sources_array
    

    
    
            
            

