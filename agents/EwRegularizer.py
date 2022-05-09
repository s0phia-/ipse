import tensorflow as tf
import numpy as np
import itertools


class BetterDefaultDict(dict):

    def __missing__(self, key):
        return self.pairwise_diff_matrix(key)

    @staticmethod
    def pairwise_diff_matrix(n):
        x = np.concatenate((np.zeros([n - 2]), [1, 1]))
        x = [list(i) for i in set(itertools.permutations(x))]
        return np.float32(x)


@tf.keras.utils.register_keras_serializable(package='Custom', name='ew')
class KerasEWRegularizer(tf.keras.regularizers.Regularizer):

    def __init__(self, reg_strength=1.5):  # pylint: disable=redefined-outer-name
        self.reg_strength = reg_strength
        self.d = BetterDefaultDict()

    def __call__(self, x):
        d = self.d[x.shape[1]]
        y = tf.linalg.matmul(d, x, transpose_b=True)
        y = tf.norm(y, 2)**2
        return self.reg_strength * y

    def get_config(self):
        return {'reg_strength': float(self.reg_strength)}
