import numpy as np
import tensorflow as tf

def unstack_on_t_axis(tensor):
    return tf.unstack(tensor, axis=1)


def sigmoid(x):
	return 1./(1 + np.exp(-x))