import tensorflow as tf

def l_1_1_norm(tf_spec_est, tf_spec_ora):
    diff = tf.math.subtract(tf_spec_est, tf_spec_ora)
    return tf.math.reduce_sum(tf.math.abs(diff))

def root_mean_square_error(tf_spec_est, tf_spec_ora):
    diff = tf.math.subtract(tf_spec_est, tf_spec_ora)
    return tf.math.sqrt(tf.reduce_mean(tf.square(diff)))

def mean_square_error(tf_spec_est, tf_spec_ora):
    diff = tf.math.subtract(tf_spec_est, tf_spec_ora)
    return tf.reduce_mean(tf.square(diff))