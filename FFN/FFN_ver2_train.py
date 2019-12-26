#ver2: 3つ入力

import time
import pprint
import sys
import IPython
import numpy as np
import tensorflow as tf
tf.reset_default_graph()
sys.path.append('../')
from model.DataProvider import DataProvider
from model.EarlyStopping import EarlyStopping
from model.NetSaver import NetSaver
from model.FFN.ffn_ver2 import FFN_ver2
from model import Loss
from model import Trainer
from model.DataArgument import DataArgument
from model import Masks
from model.STFT_Module import STFT_Module
from lib import AudioModule
from visualizer import visualize_loss
from visualizer import visualize_spec


class Train():
        def __init__(self, epoch_num=4000, batch_size=20, lr_init = 0.0001, fs = 16000, sec = 6, train_data_num=2000, valid_data_num=200, sample_len=66304):
                self.epoch_num= epoch_num
                self.batch_size = batch_size
                self.input_shape = None
                self.lr_init = lr_init
                self.train_iter = None
                self.valid_iter = None
                self.train_loss_list = []
                self.valid_loss_list = []
                self.fs = fs
                self.sec = sec
                self.train_data_num = train_data_num
                self.valid_data_num = valid_data_num
                self.sample_len = sample_len
               
                self.stft_params = {
                        "frame_length": 1024,
                        "frame_step":  256,
                        "fft_length": 1024,
                        "pad_end": False
                 }
                self.mr1_stft_params = {
                    "frame_length": 512,
                    "frame_step": 128,
                    "fft_length": 512,
                    "pad_end": False
                 }
                
                self.mr2_stft_params = {
                    "frame_length": 2048,
                    "frame_step": 512,
                    "fft_length": 2048,
                    "pad_end": False
                 }
                
                self.epsilon = 1e-4
                
        def __model(self, tf_mix, tf_target, tf_lr):
                stft_module = STFT_Module(
                        frame_length = self.stft_params["frame_length"], 
                        frame_step= self.stft_params["frame_step"], 
                        fft_length = self.stft_params["fft_length"],
                        epsilon = self.epsilon,
                        pad_end = self.stft_params["pad_end"]
                 )
                
                mr1_stft_module = STFT_Module(
                        frame_length = self.mr1_stft_params["frame_length"], 
                        frame_step= self.mr1_stft_params["frame_step"], 
                        fft_length = self.mr1_stft_params["fft_length"],
                        epsilon = self.epsilon,
                        pad_end = self.mr1_stft_params["pad_end"]
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
                tf_amp_spec_mix = stft_module.to_amp_spec(tf_spec_mix, normalize =False)
                tf_mag_spec_mix = tf.log(tf_amp_spec_mix + self.epsilon)
#                 tf_mag_spec_mix = tf.expand_dims(tf_mag_spec_mix, -1)# (Batch, Time, Freq, Channel))
#                 tf_amp_spec_mix = tf.expand_dims(tf_amp_spec_mix, -1)
                tf_f_512_mag_spec_mix = stft_module.to_F_512(tf_mag_spec_mix)
                
                 #mr1 mix data transform
                tf_mr1_spec_mix = mr1_stft_module.STFT(tf_mix)
                tf_mr1_spec_mix = tf_mr1_spec_mix[:, 1:513, :]
                tf_mr1_amp_spec_mix = stft_module.to_amp_spec(tf_mr1_spec_mix, normalize =False)
                tf_mr1_mag_spec_mix = tf.log(tf_mr1_amp_spec_mix + self.epsilon)
#                 tf_mr1_mag_spec_mix = tf.expand_dims(tf_mr1_mag_spec_mix, -1)# (Batch, Time, Freq, Channel))
                tf_mr1_f_256_mag_spec_mix = tf_mr1_mag_spec_mix[:, :, :256]
                
                #mr2 mix data transform
                #zero pad to fit stft time length 128
                mr2_zero_pad = tf.zeros_like(tf_mix)
                tf_mr2_mix = tf.concat([mr2_zero_pad[:,:384], tf_mix, mr2_zero_pad[:,:384]], axis=1)
                tf_mr2_spec_mix = mr2_stft_module.STFT(tf_mr2_mix)
                tf_mr2_amp_spec_mix = stft_module.to_amp_spec(tf_mr2_spec_mix, normalize =False)
                tf_mr2_mag_spec_mix = tf.log(tf_mr2_amp_spec_mix + self.epsilon)
#                 tf_mr2_mag_spec_mix = tf.expand_dims(tf_mr2_mag_spec_mix, -1)
                tf_mr2_mag_spec_mix = tf_mr2_mag_spec_mix[:, : ,:1024]
            
                # target data transform
                tf_spec_target = stft_module.STFT(tf_target)
                tf_amp_spec_target = stft_module.to_amp_spec(tf_spec_target, normalize=False)
#                 tf_amp_spec_target = tf.expand_dims(tf_amp_spec_target, -1)
                
                ffn_ver2 = FFN_ver2(
                        out_dim = 512,
                        h_dim = 512,
                )
            
                tf_est_masks = ffn_ver2(tf_f_512_mag_spec_mix, tf_mr1_f_256_mag_spec_mix, tf_mr2_mag_spec_mix)
                
                #F: 512  → 513
                zero_pad = tf.zeros_like(tf_mag_spec_mix)
                zero_pad = tf.expand_dims(zero_pad[:,:,1], -1)
                tf_est_masks = tf.concat( [tf_est_masks, zero_pad], 2)
                print("est_mask", tf_est_masks.shape)
                print("amp_spec_mix", tf_amp_spec_mix.shape) 
                tf_est_spec = tf.math.multiply(tf_est_masks, tf_amp_spec_mix)
                tf_loss = 10 * Loss.mean_square_error(tf_est_spec, tf_amp_spec_target)
                tf_train_step = Trainer.Adam(tf_loss, tf_lr)
                
                return tf_train_step, tf_loss, tf_amp_spec_target, tf_mag_spec_mix,   tf_spec_mix, tf_est_masks, tf_est_spec
                
        def __call__(self):                                              
                # load all train data
                provider = DataProvider()
                bass_list, drums_list, other_list, vocals_list = provider.load_all_train_data()
                # split train valid
                train_bass_list,    valid_bass_list = provider.split_to_train_valid(bass_list)
                train_drums_list, valid_drums_list = provider.split_to_train_valid(drums_list)
                train_other_list,   valid_other_list = provider.split_to_train_valid(other_list)
                train_vocals_list,  valid_vocals_list = provider.split_to_train_valid(vocals_list)
                # define model
                tf_lr = tf.placeholder(tf.float32) # learning rate
                tf_mix = tf.placeholder(tf.float32, (self.batch_size, self.sample_len)) #Batch, Sample
                tf_target = tf.placeholder(tf.float32, (self.batch_size, self.sample_len)) #Batch,Sample
                
                tf_train_step, tf_loss , tf_target_spec, tf_mag_mix_spec, tf_ori_mix_spec, tf_est_masks, tf_est_spec = self.__model(tf_mix, tf_target, tf_lr)
                
                # GPU config
                config = tf.ConfigProto(
                        gpu_options=tf.GPUOptions(
                                visible_device_list='1', # specify GPU number
                                allow_growth = True
                        )
                )
                with tf.Session(config = config) as sess:
                        init = tf.global_variables_initializer()  
                        sess.run(init)
                        print("Start Training")
                        net_saver = NetSaver(saver_folder_name='FFN_ver2',  saver_file_name='ffn_ver2')
                        early_stopping = EarlyStopping()
                        for epoch in range(self.epoch_num):
                                sys.stdout.flush()
                                print('epoch:' + str(epoch))
                                start = time.time()

                                train_data_argument = DataArgument(self.fs, self.sec, self.train_data_num)
                                train_arg_bass_array = train_data_argument(train_bass_list)
                                train_arg_drums_array = train_data_argument(train_drums_list)                                
                                train_arg_other_array = train_data_argument(train_other_list)
                                train_arg_vocals_array = train_data_argument(train_vocals_list)
                                
                                valid_data_argument = DataArgument(self.fs, self.sec, self.valid_data_num)
                                valid_arg_bass_array = valid_data_argument(valid_bass_list)
                                valid_arg_drums_array = valid_data_argument(valid_drums_list)
                                valid_arg_other_array = valid_data_argument(valid_other_list)
                                valid_arg_vocals_array = valid_data_argument(valid_vocals_list)  
                                
                                self.train_iter = int(len(train_arg_bass_array) / self.batch_size)
                                self.valid_iter = int(len(valid_arg_bass_array) / self.batch_size)
                                # mixing
                                train_mixed_array = AudioModule.mixing(
                                                                                    train_arg_bass_array,
                                                                                    train_arg_drums_array,
                                                                                    train_arg_other_array,
                                                                                    train_arg_vocals_array
                                                                            )
                                train_target_array = train_arg_vocals_array
                                
                                valid_mixed_array = AudioModule.mixing(
                                                                                    valid_arg_bass_array,
                                                                                    valid_arg_drums_array,
                                                                                    valid_arg_other_array,
                                                                                    valid_arg_vocals_array
                                                                            )
                                valid_target_array = valid_arg_vocals_array
#                                
                                # training
                                
                                tf.keras.backend.set_learning_phase(1)
                                for train_time in range(self.train_iter):
                                    sess.run(tf_train_step, feed_dict = {
                                           tf_mix: train_mixed_array[train_time*self.batch_size:(train_time+1)*self.batch_size, :self.sample_len],
                                           tf_target: train_target_array[train_time*self.batch_size:(train_time+1)*self.batch_size, :self.sample_len],
                                           tf_lr: self.lr_init
                                        }
                                     )
                            
                                tmp_valid_loss_list = [] 
                                tf.keras.backend.set_learning_phase(0) 
                                for valid_time in range(self.valid_iter):                
                                    valid_loss = sess.run(tf_loss, feed_dict = {
                                               tf_mix: valid_mixed_array[valid_time*self.batch_size:(valid_time+1)*self.batch_size, :self.sample_len],
                                               tf_target: valid_target_array[valid_time*self.batch_size:(valid_time+1)*self.batch_size, :self.sample_len],
                                               tf_lr:  0.
                                            }
                                         )
                                    tmp_valid_loss_list.append(valid_loss)

                                self.valid_loss_list.append(np.mean(tmp_valid_loss_list))
                            
                                vmin = -70
                                vmax = 0
                                
                                target_spec, mag_mix_spec, ori_spec_mix, est_mask, est_spec = sess.run([tf_target_spec, tf_mag_mix_spec , tf_ori_mix_spec, tf_est_masks, tf_est_spec], feed_dict ={
                                    tf_mix: train_mixed_array[:self.batch_size, :self.sample_len],
                                    tf_target: train_target_array[:self.batch_size, :self.sample_len],
                                    tf_lr: 0.
                                })
                    
#                                 est_mask = np.squeeze(est_mask, axis=-1)
#                                 target_spec = np.squeeze(target_spec, axis=-1)
#                                 mag_mix_spec = np.squeeze(mag_mix_spec, axis=-1)
#                                 est_spec = np.squeeze(est_spec, axis=-1)
                                print("original spec mix")
                                visualize_spec.plot_spec(ori_spec_mix[0], self.fs, self.sec, vmax, vmin)
                                print("magnitude spec mix")
                                visualize_spec.plot_log_spec(mag_mix_spec[0], self.fs, self.sec, 10, -10)
                                print("target spec")
                                visualize_spec.plot_spec(target_spec[0], self.fs, self.sec, vmax, vmin)
                                print("est mask")
                                visualize_spec.plot_log_spec(est_mask[0], self.fs, self.sec,  1, 0)
                                print("est spec")
                                visualize_spec.plot_spec(est_spec[0], self.fs, self.sec,  vmax, vmin)
                
                                visualize_loss.plot_loss(self.valid_loss_list)
                                end = time.time()
                                print(' excute time', end - start)
                                if epoch ==  3999:
                                    net_saver(sess, step=epoch)
                        

if __name__ == '__main__':
    train = Train()
    train()

