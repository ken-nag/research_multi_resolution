#ver3: 短い窓を補助情報
import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization,  LeakyReLU, Conv2DTranspose, concatenate
from tensorflow.keras.activations import sigmoid

class Conv_FFN_ver3():
    def __init__(self, input_shape,mr1_input_shape, mr2_input_shape, h_dim, out_dim):
        self.kernel_size = (5,5)
        self.stride = (2,4)
        self.mr1_stride = (4,2)
        self.mr2_stride = (1, 8)
        self.de_stride = (2,1)
        self.leakiness = 0.2
        
        with tf.variable_scope("Conv_FFN_ver3"):
            #mr1: t 512 to 256
            self.mr1_conv1 = Conv2D(2, self.kernel_size, self.mr1_stride, input_shape=mr1_input_shape, padding='same')
            self.mr1_Bnorm1 = BatchNormalization()
            #mr2:
            self.mr2_conv1 = Conv2D(2, self.kernel_size, self.mr2_stride, input_shape=mr2_input_shape, padding='same')
            self.mr2_Bnorm1 = BatchNormalization()
            #
            self.conv1  = Conv2D(2, self.kernel_size, self.stride, input_shape=input_shape, padding='same')
            self.Bnorm1 = BatchNormalization()
            
            self.dense1 = Dense(h_dim, activation='relu')
            self.dense2 = Dense(h_dim, activation='relu')
            self.dense3 = Dense(h_dim, activation='relu')
            self.dense4 = Dense(h_dim, activation='relu')
            self.dense5 = Dense(out_dim, activation='relu')
            self.deconv1  = Conv2DTranspose(1, self.kernel_size, self.de_stride, input_shape=(128, 512, 1),padding='same')
        
    def to_input_shape(self, tf_X):
        tmp_tf_X = tf.concat([ tf_X[:,:-4,:,:], tf_X[:,1:-3,:,:], tf_X[:,2:-2,:,:], tf_X[:,3:-1,:,:], tf_X[:,4:,:,:] ], axis=2)
        return  tf.concat([tmp_tf_X[:,:,:,0], tmp_tf_X[:,:,:,1], tmp_tf_X[:,:,:,2], tmp_tf_X[:,:,:,4], tmp_tf_X[:,:,:,5] ], axis=2)
    
    def zero_pad(self, tf_X):
        zero_mat = tf.zeros([2,512], dtype=tf.dtypes.float32)
        return tf.concat([zero_mat, tf_X, zero_mat], axis=0)
          
    def __call__(self, tf_X, tf_mr1_X, tf_mr2_X):
        mask_list = []
        h1 = LeakyReLU(alpha = self.leakiness)(self.Bnorm1(self.conv1(tf_X)))
        mr1_h1 = LeakyReLU(alpha=self.leakiness)(self.mr1_Bnorm1(self.mr1_conv1(tf_mr1_X)))
        mr2_h1 = LeakyReLU(alpha=self.leakiness)(self.mr2_Bnorm1(self.mr2_conv1(tf_mr2_X)))
        
        h1_reshaped = self.to_input_shape(concatenate([h1, mr1_h1, mr2_h1]))
        print("h1_reshaped:", h1_reshaped.shape)
        
        for batch_num in range(h1_reshaped.shape[0]):
            h2 = self.dense1(h1_reshaped[batch_num, :, :])
            h3 = self.dense2(h2)
            h4 = self.dense3(h3)
            h5 = self.dense4(h4)
            h5 = self.zero_pad(h5)
            mask_list.append(h5)
        h6 = tf.convert_to_tensor(mask_list, dtype=tf.float32)
        h6 = tf.expand_dims(h6, -1)
        dh1 = sigmoid(self.deconv1(h6))
        print("dh1:", dh1.shape)
        return dh1
   
        