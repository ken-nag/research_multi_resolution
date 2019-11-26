import tensorflow as tf
import sys
import os
sys.path.append('../')

class NetSaver():
    def __init__(self, saver_folder_name, saver_file_name):
        self.results_model_path = '../results/model/'
        self.saver_folder_name = saver_folder_name + '/'
        self.saver_file_name = saver_file_name
        
        saver_folder_path = self.results_model_path + saver_folder_name
        self.__exist_or_create(saver_folder_path)
    
    def __call__(self, sess, step):
        saver = tf.train.Saver(max_to_keep=None)
        save_file = self.results_model_path + self.saver_folder_name + self.saver_file_name
        saver.save(sess, save_file + '_{}.ckpt'.format(step))
        
        
    def __exist_or_create(self, folder_path):
        if(os.path.isdir(folder_path)==False):
            os.mkdir(folder_path)