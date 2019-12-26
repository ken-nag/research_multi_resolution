import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization,  LeakyReLU, Conv2DTranspose, concatenate
from tensorflow.keras.activations import sigmoid

class FFN():
    def __init__(self, h_dim, out_dim): 
        with tf.variable_scope("FFN_ver2"):
            self.dense1 = Dense(h_dim, activation='relu')
            self.dense2 = Dense(h_dim, activation='relu')
            self.dense3 = Dense(h_dim, activation='relu')
            self.dense4 = Dense(h_dim, activation='relu')
            self.dense5 = Dense(out_dim, activation='sigmoid')
           
    def zero_pad(self, tf_X):
        zero_mat = tf.zeros([2,512], dtype=tf.dtypes.float32)
        return tf.concat([zero_mat, tf_X, zero_mat], axis=0)
             
    def to_input_shape(self, tf_X, tf_mr1_X, tf_mr2_X):
        tmp_tf_mr1_X = tf.concat([tf_mr1_X[:, ::2, :], tf_mr1_X[:,1::2,:]], axis=2)
        tmp_tf_mr2_X = tf.reshape(tf.concat([tf_mr2_X, tf_mr2_X], axis=2), [-1, 256, 1024])
        tmp_input = tf.concat([tf_X, tmp_tf_mr1_X, tmp_tf_mr2_X], axis=2)
        return tf.concat([tmp_input[:,:-4,:], tmp_input[:,1:-3,:], tmp_input[:,2:-2,:], tmp_input[:,3:-1,:], tmp_input[:,4:,:]], axis=2)
        
        
    def __call__(self, tf_X):
        mask_list = []
        h1 = self.to_input_shape(tf_X, tf_mr1_X, tf_mr2_X) 
        for batch_num in range(h1.shape[0]):
            h2 = self.dense1(h1[batch_num, :, :])
            h3 = self.dense2(h2)
            h4 = self.dense3(h3)
            h5 = self.dense4(h4)
            h5 = self.zero_pad(h5)
            mask_list.append(h5)
        h6 = tf.convert_to_tensor(mask_list, dtype=tf.float32)
        print("h6:", h6.shape)
        return h6
   
        
        