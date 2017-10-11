import tensorflow as tf
import numpy as np
import inspect
from collections import deque
import random


def make_weights(rows, cols):
    weights = tf.Variable(tf.random_uniform([rows, cols], -0.001, 0.001))
    bias = tf.Variable(tf.constant(np.random.random(), shape=[1, cols]))
    return weights, bias


def compute_next_layer(input_layer, weights, bias, activation=tf.nn.relu):
    h = tf.matmul(input_layer, weights) + bias
    output = activation(h) if activation else tf.identity(h)
    return output


def save_args(init):
    def save(*args, **kwargs):
        bound = inspect.signature(init).bind(*args, **kwargs).arguments
        instance = None
        for k, v in bound.items():
            if instance is None:
                instance = v
            else:
                setattr(instance, k, v)
        return init(*args, **kwargs)
    return save


class Normalizer:
    @save_args
    def __init__(self, xmin, xmax):
        self.xmin = xmin
        self.xmax = xmax

    def is_inside(self, x):
        return self.xmin <= x <= self.xmax

    def normalize(self, x):
        return 2 * (x - self.xmin) / (self.xmax - self.xmin) - 1

    def denormalize(self, x):
        return self.xmin + (self.xmax - self.xmin) * (x + 1) / 2

    def __call__(self, x):
        return self.normalize(x)


class KMostRecent:
    @save_args
    def __init__(self, max_size):
        self.buffer = deque()
    
    def add(self, thing):
        self.buffer.appendleft(thing)
        if len(self.buffer) > self.max_size:
            self.buffer.pop()

    def random_sample(self, size):
        if size < len(self.buffer):
            return random.sample(self.buffer, size)
        else:
            return list(self.buffer)

    def is_full(self):
        return len(self.buffer) == self.max_size
