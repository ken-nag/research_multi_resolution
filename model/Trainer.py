import tensorflow as tf

def Adam(loss, tf_lr):
    train_step = tf.train.AdamOptimizer(tf_lr).minimize(loss)
    return train_step


def SGD(loss, tf_lr):
    train_step = tf.train.GradientDescentOptimizer(tf_lr).minimize(loss)
    return train_step