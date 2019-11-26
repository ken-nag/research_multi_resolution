import numpy as np

def si_sdr(spec_ora, spec_est, epsilon):
    alpha = tf.dot(spec_ora, spec_est) / tf.dot(spec_ora, spec_ora)