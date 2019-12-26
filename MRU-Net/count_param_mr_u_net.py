import time
import pprint
import sys
import IPython
import mir_eval
import numpy as np
import tensorflow as tf
tf.reset_default_graph()
sys.path.append('../')
from model.DataProvider import DataProvider
from model.EarlyStopping import EarlyStopping
from model.NetSaver import NetSaver
from model.MRUNet.MRUNet_ver5 import MRUNet_ver5
from model import Loss
from model import Trainer
from model.DataArgument import DataArgument
from model import Masks
from model.STFT_Module import STFT_Module
from lib import AudioModule
from visualizer import visualize_loss
from visualizer import visualize_spec


class Test():
        def __init__(self, epoch_num=1, batch_size=10, fs = 16000, sec = 18, test_data_num=10000, sample_len=66304):
                self.epoch_num= epoch_num
                self.batch_size = batch_size
                self.input_shape = None
                self.test_iter = None
                self.fs = fs
                self.sec = sec
                self.test_data_num = test_data_num
                self.stft_params = {
                        "frame_length": 1024,
                        "frame_step": 256,
                        "fft_length": 1024,
                        "pad_end": False
                }

                self.mr2_stft_params = {
                    "frame_length": 2048,
                    "frame_step": 512,
                    "fft_length": 2048,
                    "pad_end": False
                 }
                
                self.est_audio_list = []
                self.sdr_list = []
                self.sir_list = []
                self.sar_list = []
                self.sample_len = sample_len
                self.epsilon = 1e-4
        
        def expand_channel(self, tf_X):
                return tf.expand_dims(tf_X, -1)
            
        def __model(self, tf_mix):
                 # define model flow
                # stft
                stft_module = STFT_Module(
                        frame_length = self.stft_params["frame_length"], 
                        frame_step= self.stft_params["frame_step"], 
                        fft_length = self.stft_params["fft_length"],
                        epsilon = self.epsilon,
                        pad_end = self.stft_params["pad_end"]
                )
                
                mr2_stft_module = STFT_Module(
                        frame_length = self.mr2_stft_params["frame_length"], 
                        frame_step= self.mr2_stft_params["frame_step"], 
                        fft_length = self.mr2_stft_params["fft_length"],
                        epsilon = self.epsilon,
                        pad_end = self.mr2_stft_params["pad_end"]
                )
                
                
                # mix data transform
                tf_spec_mix = stft_module.STFT(tf_mix)
                tf_phase_mix = tf.sign(tf_spec_mix)
                tf_phase_mix = self.expand_channel(tf_phase_mix)
                tf_amp_spec_mix = stft_module.to_amp_spec(tf_spec_mix, normalize =False)
                tf_mag_spec_mix = tf.log(tf_amp_spec_mix + self.epsilon)
                tf_mag_spec_mix = tf.expand_dims(tf_mag_spec_mix, -1)# (Batch, Time, Freq, Channel))
                tf_amp_spec_mix = tf.expand_dims(tf_amp_spec_mix, -1)
                tf_f_512_mag_spec_mix = stft_module.to_F_512(tf_mag_spec_mix)
                               
                 #mr2 mix data transform
                #zero pad to fit stft time length 128
                mr2_zero_pad = tf.zeros_like(tf_mix)
                tf_mr2_mix = tf.concat([mr2_zero_pad[:,:384], tf_mix, mr2_zero_pad[:,:384]], axis=1)
                tf_mr2_spec_mix = mr2_stft_module.STFT(tf_mr2_mix)
                tf_mr2_amp_spec_mix = stft_module.to_amp_spec(tf_mr2_spec_mix, normalize =False)
                tf_mr2_mag_spec_mix = tf.log(tf_mr2_amp_spec_mix + self.epsilon)
                tf_mr2_mag_spec_mix = tf.expand_dims(tf_mr2_mag_spec_mix, -1)
                tf_mr2_mag_spec_mix = tf_mr2_mag_spec_mix[:, : ,:1024,:]
            
                                 
                mr_u_net_ver5 = MRUNet_ver5(
                        input_shape = (
                                tf_f_512_mag_spec_mix.shape[1:]
                        ),
                        mr2_input_shape = (
                              tf_mr2_mag_spec_mix.shape[1:]
                        )
                )
            
                tf_est_masks = mr_u_net_ver5(tf_f_512_mag_spec_mix, tf_mr2_mag_spec_mix)
                
                
                #F: 512  → 513
                zero_pad = tf.zeros_like(tf_mag_spec_mix)
                zero_pad = tf.expand_dims(zero_pad[:,:,1,:], -1)
                tf_est_masks = tf.concat( [tf_est_masks, zero_pad], 2)
                tf_est_spec = tf.math.multiply(tf_est_masks, tf_amp_spec_mix)
                tf_est_source_spec = tf.math.multiply(tf.complex(tf_est_spec, 0.), tf_phase_mix)
                tf_est_source_spec = tf.squeeze(tf_est_source_spec, axis=-1)                
                est_source = stft_module.ISTFT(tf_est_source_spec)
                return est_source
                
        def __call__(self):                                              
                # load all train data
                provider = DataProvider()
                test_bass_list, test_drums_list, test_other_list, test_vocals_list = provider.load_all_test_data()
                # define model
                tf_mix = tf.placeholder(tf.float32, (None, self.sample_len)) #Batch, Sample
                tf_est_source = self.__model(tf_mix)
                
                # GPU config
                config = tf.ConfigProto(
                        gpu_options=tf.GPUOptions(
                                visible_device_list='0', # specify GPU number
                                allow_growth = True
                        )
                )
                
                saver = tf.train.import_meta_graph('./../results/model/MRUNet_ver5/mr_u_net_ver5_1998.ckpt.meta')
                with tf.Session(config = config) as sess:
                        saver.restore(sess, './../results/model/MRUNet_ver5/mr_u_net_ver5_1998.ckpt')
                        
                        total_parameters = 0
                        parameters_string = ""

                        for variable in tf.trainable_variables():
                              shape = variable.get_shape()
                              variable_parameters = 1
                              for dim in shape:
                                    variable_parameters *= dim.value
                              total_parameters += variable_parameters
                              if len(shape) == 1:
                                    parameters_string += ("%s %d, " % (variable.name, variable_parameters))
                              else:
                                    parameters_string += ("%s %s=%d, " % (variable.name, str(shape), variable_parameters))

                        print(parameters_string)
                        print("Total %d variables, %s params" % (len(tf.trainable_variables()), "{:,}".format(total_parameters)))

if __name__ == '__main__':
    test = Test()
    est_list, target_list, mixed_list = test()
    file_path = './../results/audio/MRUNet/singing_voice_separation/'
#    AudioModule.to_pickle(est_list, file_path + 'est_list')