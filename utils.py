import tensorflow as tf
import numpy as np
import inspect
from collections import deque
import random


def make_weights(rows, cols, mag=0.1):
    weights = tf.Variable(tf.random_uniform([rows, cols], -mag, mag))
    bias = tf.Variable(tf.constant(mag * (2 * np.random.random() - 1), shape=[1, cols]))
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


class OrnsteinUhlenbeckProcess:
    # https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process
    # implemented as detailed in https://math.stackexchange.com/a/1288406/99169
    def __init__(self, theta, mu, sigma, x0, dt):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.x0 = x0
        self.dt = dt
        self.n = 0
        self.last = self.x0

    def get_noise(self):
        self.last += (
            self.theta * (self.mu - self.last) * self.dt
            + self.sigma * np.sqrt(self.dt) * np.random.normal()
        )
        return self.last

class MultidimensionalOrnsteinUhlenbeckProcess:
    def __init__(self, count, theta, mu, sigma, x0, dt):
        self.noise_processes = [
            OrnsteinUhlenbeckProcess(theta, mu, sigma, x0, dt)
            for _ in range(count)
        ]

    def get_noise(self):
        return [p.get_noise() for p in self.noise_processes]
