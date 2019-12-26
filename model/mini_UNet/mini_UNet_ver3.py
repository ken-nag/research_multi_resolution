#ver3
#入力はmr1, mr2の3つ
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv2DTranspose,BatchNormalization, Dropout, LeakyReLU, ReLU, concatenate, Input
from tensorflow.keras.activations import sigmoid
class mini_UNet_ver3():
    def __init__(self, input_shape, mr1_input_shape, mr2_input_shape):
        with tf.variable_scope("U-Net"):
            self.kernel_size  = (5,5) # (5,5)
            self.stride       = (2,2)
            self.leakiness    = 0.2
            self.dropout_rate = 0.5
            # stride for mr1
            self.t_mr1_stride = (2,1)
            self.f_mr1_stride = (1,2)
             # stride for mr2
            self.f_mr2_stride = (1,8)
            
            #mr1: t 512 to 256
            self.mr1_conv1 = Conv2D(4, self.kernel_size, self.t_mr1_stride, input_shape=mr1_input_shape, padding='same')
            self.mr1_Bnorm1 = BatchNormalization()
            
            #mr2:
            self.mr2_conv1 = Conv2D(8, self.kernel_size, self.f_mr2_stride, input_shape=mr2_input_shape, padding='same')
            self.mr2_Bnorm1 = BatchNormalization()
            
            # endcoder
            self.conv1  = Conv2D(4, self.kernel_size, self.f_mr1_stride, input_shape=input_shape, padding='same')
            self.Bnorm1 = BatchNormalization()
            self.conv2  = Conv2D(8, self.kernel_size, self.stride, padding='same')
            self.Bnorm2 = BatchNormalization()
            self.conv3  = Conv2D(32, self.kernel_size, self.stride, padding='same')
            self.Bnorm3 = BatchNormalization()
            # decoder
            self.deconv1  = Conv2DTranspose(16, self.kernel_size, self.stride, padding='same')
            self.deBnorm1 = BatchNormalization()
            self.Dropout1 = Dropout(rate = self.dropout_rate)
            self.deconv2  = Conv2DTranspose(8, self.kernel_size, self.stride, padding='same')
            self.deBnorm2 = BatchNormalization()
            self.deconv3  = Conv2DTranspose(1, self.kernel_size, self.f_mr1_stride, padding='same')
            self.deBnorm3 = BatchNormalization()

    def __call__(self, tf_X, tf_mr1_X, tf_mr2_X):
        mr1_h1 = LeakyReLU(alpha=self.leakiness)(self.mr1_Bnorm1(self.mr1_conv1(tf_mr1_X)))
        mr2_h1 = LeakyReLU(alpha=self.leakiness)(self.mr2_Bnorm1(self.mr2_conv1(tf_mr2_X)))
        print("mr_h1", mr1_h1.shape)
        h1 = LeakyReLU(alpha = self.leakiness)(self.Bnorm1(self.conv1(tf_X)))
        print(h1.shape)
        h2 = LeakyReLU(alpha = self.leakiness)(self.Bnorm2(self.conv2(concatenate([h1, mr1_h1]))))
        print(h2.shape)
        h3 = LeakyReLU(alpha = self.leakiness)(self.Bnorm3(self.conv3(concatenate([h2, mr2_h1]))))
        print(h3.shape)
        dh1 = ReLU()(self.Dropout1(self.deBnorm1(self.deconv1(h3))))
        print(dh1.shape)
        dh2 = ReLU()(self.deBnorm2(self.deconv2(concatenate([dh1, h2]))))
        print(dh2.shape)
        dh3 = ReLU()(self.deBnorm3(self.deconv3(concatenate([dh2, h1]))))
        print(dh3.shape)

        return dh3, h1, h2, h3, dh1, dh2
