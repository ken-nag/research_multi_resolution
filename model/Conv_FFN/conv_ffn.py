import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization,  LeakyReLU, Conv2DTranspose
from tensorflow.keras.activations import sigmoid

class Conv_FFN():
    def __init__(self, input_shape, h_dim, out_dim):
        self.kernel_size = (5,5)
        self.stride = (2,2)
        self.f_stride = (1, 2)
        self.de_stride = (2,1)
        self.leakiness = 0.2
        
        with tf.variable_scope("Conv_FFN"):
            self.conv1  = Conv2D(6, self.kernel_size, self.f_stride, input_shape=input_shape, padding='same')
            self.Bnorm1 = BatchNormalization()
            self.conv2 =  Conv2D(6, self.kernel_size, self.stride, padding='same')
            self.Bnorm2 = BatchNormalization()
            
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
          
    def __call__(self, tf_X):
        mask_list = []
        h1 = LeakyReLU(alpha = self.leakiness)(self.Bnorm1(self.conv1(tf_X)))
        h2 = LeakyReLU(alpha = self.leakiness)(self.Bnorm2(self.conv2(h1)))
        print("h2:", h2.shape)
        h3 = self.to_input_shape(h2)
        print("h3:", h3.shape)
        
        for batch_num in range(h3.shape[0]):
            h4 = self.dense1(h3[batch_num, :, :])
            h5 = self.dense2(h4)
            h6 = self.dense3(h5)
            h7 = self.dense4(h6)
            h8 = self.zero_pad(h7)
            mask_list.append(h8)
        h9 = tf.convert_to_tensor(mask_list, dtype=tf.float32)
        h10 = tf.expand_dims(h9, -1)
        dh1 = sigmoid(self.deconv1(h10))
        print("dh1:", dh1.shape)
        return dh1
   
        