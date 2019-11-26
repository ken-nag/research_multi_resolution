import tensorflow as tf

def iaf(tf_mix_spec, tf_target_spec, epsilon):
    bound = tf.constant(1.0, dtype=tf.float32)
    ora_amp_mask = tf.abs(tf_target_spec) / (tf.abs(tf_mix_spec) + epsilon)
    return tf.math.minimum(ora_amp_mask, bound)