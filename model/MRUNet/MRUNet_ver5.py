#ver5
#入力を2種類
#mr2: 窓幅2048
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv2DTranspose,BatchNormalization, Dropout, LeakyReLU, ReLU, concatenate, Input
from tensorflow.keras.activations import sigmoid
class MRUNet_ver5():
    def __init__(self, input_shape, mr2_input_shape):
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
            
            #mr2:
            self.mr2_conv1 = Conv2D(16, self.kernel_size, self.f_mr2_stride, input_shape=mr2_input_shape, padding='same')
            self.mr2_Bnorm1 = BatchNormalization()
            # endcoder
            self.conv1  = Conv2D(16, self.kernel_size, self.f_mr1_stride, input_shape=input_shape, padding='same')
            self.Bnorm1 = BatchNormalization()
            self.conv2  = Conv2D(16, self.kernel_size, self.stride, padding='same')
            self.Bnorm2 = BatchNormalization()
            self.conv3  = Conv2D(64, self.kernel_size, self.stride, padding='same')
            self.Bnorm3 = BatchNormalization()
            self.conv4  = Conv2D(128, self.kernel_size, self.stride, padding='same')
            self.Bnorm4 = BatchNormalization()
            self.conv5  = Conv2D(256, self.kernel_size, self.stride, padding='same')
            self.Bnorm5 = BatchNormalization()
            self.conv6  = Conv2D(512, self.kernel_size, self.stride, padding='same')
            self.Bnorm6 = BatchNormalization()
            # decoder
            self.deconv1  = Conv2DTranspose(256, self.kernel_size, self.stride, padding='same')
            self.deBnorm1 = BatchNormalization()
            self.Dropout1 = Dropout(rate = self.dropout_rate)
            self.deconv2  = Conv2DTranspose(128, self.kernel_size, self.stride, padding='same')
            self.deBnorm2 = BatchNormalization()
            self.Dropout2 = Dropout(rate = self.dropout_rate)
            self.deconv3  = Conv2DTranspose(64, self.kernel_size, self.stride, padding='same')
            self.deBnorm3 = BatchNormalization()
            self.Dropout3 = Dropout(rate = self.dropout_rate)
            self.deconv4  = Conv2DTranspose(32, self.kernel_size, self.stride, padding='same')
            self.deBnorm4 = BatchNormalization()
            self.deconv5  = Conv2DTranspose(16, self.kernel_size, self.stride, padding='same')
            self.deBnorm5 = BatchNormalization()
            self.deconv6  = Conv2DTranspose(1, self.kernel_size, self.f_mr1_stride, padding='same')



    def __call__(self, tf_X, tf_mr2_X):
        mr2_h1 = LeakyReLU(alpha=self.leakiness)(self.mr2_Bnorm1(self.mr2_conv1(tf_mr2_X)))
        print("mr2_h1", mr2_h1.shape)
        h1 = LeakyReLU(alpha = self.leakiness)(self.Bnorm1(self.conv1(tf_X)))
        print("h1", h1.shape)
        print(h1.shape)
        h2 = LeakyReLU(alpha = self.leakiness)(self.Bnorm2(self.conv2(h1)))
        print(h2.shape)
        h3 = LeakyReLU(alpha = self.leakiness)(self.Bnorm3(self.conv3(concatenate([h2, mr2_h1]))))
        print(h3.shape)
        h4 = LeakyReLU(alpha = self.leakiness)(self.Bnorm4(self.conv4(h3)))
        print(h4.shape)
        h5 = LeakyReLU(alpha = self.leakiness)(self.Bnorm5(self.conv5(h4)))
        print(h5.shape)
        h6 = LeakyReLU(alpha = self.leakiness)(self.Bnorm6(self.conv6(h5)))
        print(h6.shape)
        dh1 = ReLU()(self.Dropout1(self.deBnorm1(self.deconv1(h6))))
        print(dh1.shape)
        dh2 = ReLU()(self.Dropout2(self.deBnorm2(self.deconv2(concatenate([dh1, h5])))))
        print(dh2.shape)
        dh3 = ReLU()(self.Dropout3(self.deBnorm3(self.deconv3(concatenate([dh2, h4])))))
        print(dh3.shape)
        dh4 = ReLU()(self.deBnorm4(self.deconv4(concatenate([dh3, h3]))))
        print(dh4.shape)
        dh5 = ReLU()(self.deBnorm5(self.deconv5(concatenate([dh4, h2]))))
        print(dh5.shape)
        dh6 = sigmoid(self.deconv6(concatenate([dh5, h1])))
        print(dh6.shape)

        return dh6
