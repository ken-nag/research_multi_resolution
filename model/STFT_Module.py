import tensorflow as tf
from tensorflow.contrib.signal import stft, inverse_stft, hann_window

class STFT_Module():
    def __init__(self, frame_length, frame_step, fft_length, epsilon, pad_end=True):
            self.frame_length = frame_length
            self.frame_step = frame_step
            self.fft_length = fft_length
            self.pad_end = pad_end
            self.epsilon = epsilon

    def STFT(self, tf_x):
            specs =  tf.signal.stft(
                              signals = tf_x,
                              frame_length = self.frame_length,
                              frame_step = self.frame_step,
                              fft_length = self.fft_length,
                              window_fn = tf.signal.hann_window,
                              pad_end = self.pad_end,
                              name=None
                          )
            return specs
        
    def to_magnitude_spec(self, tf_X, normalize=True):
            amp_spec = tf.abs(tf_X)
            if normalize:
                normalized_amp_spec =  self.__normalize(amp_spec)
                return tf.log(normalized_amp_spec + self.epsilon)
            else:
                return tf.log(amp_spec + self.epsilon)
                
    def to_amp_spec(self, tf_X, normalize=True):
            abs_spec = tf.abs(tf_X)
            if normalize:
                return self.__normalize(abs_spec)
            else:
                return abs_spec
            
    def ISTFT(self, tf_X):
            waves = tf.signal.inverse_stft(
                             stfts = tf_X,
                             frame_length = self.frame_length,
                             frame_step = self.frame_step,
                             fft_length = self.fft_length,
                             window_fn = tf.contrib.signal.inverse_stft_window_fn(self.frame_step),
                             name=None
                         )
            return waves
    
    
    def to_T_256(self, tf_X):
            return tf_X[:, :256, :]
       
    def to_F_512(self, tf_X):
            return tf_X[:, :, :512]
                
    # def get_phase(self, tf_x):
    #        return 
    
    def __normalize(self, tf_mag_X):
            maximums = tf.math.reduce_max(tf_mag_X, axis=(1,2), keep_dims=True)
            return tf.div(tf_mag_X, (maximums+self.epsilon))

