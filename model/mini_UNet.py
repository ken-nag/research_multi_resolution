#drop out はdh1のみ
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv2DTranspose,BatchNormalization, Dropout, LeakyReLU, ReLU, concatenate, Input
from tensorflow.keras.activations import sigmoid
class mini_UNet():
    def __init__(self, input_shape):
        with tf.variable_scope("U-Net"):
            self.kernel_size  = (5,5) # (5,5)
            self.stride       = (2,2)
            self.leakiness    = 0.2
            self.dropout_rate = 0.5
            # endcoder
            self.conv1  = Conv2D(8, self.kernel_size, self.stride, input_shape=input_shape, padding='same')
            self.Bnorm1 = BatchNormalization()
            self.conv2  = Conv2D(16, self.kernel_size, self.stride, padding='same')
            self.Bnorm2 = BatchNormalization()
            self.conv3  = Conv2D(32, self.kernel_size, self.stride, padding='same')
            self.Bnorm3 = BatchNormalization()
            # decoder
            self.deconv1  = Conv2DTranspose(16, self.kernel_size, self.stride, padding='same')
            self.deBnorm1 = BatchNormalization()
            self.Dropout1 = Dropout(rate = self.dropout_rate)
            self.deconv2  = Conv2DTranspose(8, self.kernel_size, self.stride, padding='same')
            self.deBnorm2 = BatchNormalization()
            self.deconv3  = Conv2DTranspose(1, self.kernel_size, self.stride, padding='same')
            self.deBnorm3 = BatchNormalization()

    def __call__(self, tf_X):
        h1 = LeakyReLU(alpha = self.leakiness)(self.Bnorm1(self.conv1(tf_X)))
        print(h1.shape)
        h2 = LeakyReLU(alpha = self.leakiness)(self.Bnorm2(self.conv2(h1)))
        print(h2.shape)
        h3 = LeakyReLU(alpha = self.leakiness)(self.Bnorm3(self.conv3(h2)))
        print(h3.shape)
        dh1 = ReLU()(self.Dropout1(self.deBnorm1(self.deconv1(h6))))
        print(dh1.shape)
        dh2 = ReLU()(self.deBnorm2(self.deconv2(concatenate([dh1, h5]))))
        print(dh2.shape)
        dh3 = ReLU()(self.deBnorm3(self.deconv3(concatenate([dh2, h4]))))
        print(dh3.shape)

        return dh3, h1, h2, h3, dh1, dh2
