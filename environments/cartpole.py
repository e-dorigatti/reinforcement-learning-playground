from __future__ import print_function
import numpy as np
from math import sin, cos, pi
from utils import save_args
import matplotlib.pyplot as plt
from learning_task import Environment
from utils import Normalizer


class PendulumDynamics:
    m = 0.2     # mass of the pendulum (kg)
    M = 1.0     # mass of the cart (kg)
    l = 0.5     # length of the rod (m)
    g = 9.81    # gravitational acceleration (m/s^2)
    b = 5.0     # friction coefficient (?)
    dt = 0.02   # simulation delta time (s)

    def __init__(self, x, xdot, theta, thetadot):
        self.x = x
        self.xdot = xdot
        self.theta = theta
        self.thetadot = thetadot

    def step_simulate(self, force):
        xacc_numerator = (force -
                          self.m * self.l * self.thetadot**2 * sin(self.theta) +
                          self.m * self.g * sin(self.theta) * cos(self.theta) -
                          self.b * self.xdot)

        xacc_denominator = self.M + self.m - self.m * cos(self.theta)**2
        xacc = xacc_numerator / xacc_denominator

        theta_acc = (xacc*cos(self.theta) + self.g * sin(self.theta)) / self.l

        self.xdot = self.xdot + xacc * self.dt
        self.x += self.xdot * self.dt

        self.thetadot = self.thetadot + theta_acc * self.dt
        self.theta += self.thetadot * self.dt

        if self.theta > pi / 2:
            self.theta -= pi
        elif self.theta < -pi / 2:
            self.theta += pi 

    @property
    def state(self):
        return self.x, self.xdot, self.theta, self.thetadot


def to_deg(rad):
    return rad*180/pi


def plot(time, positions, angles):
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.title('Cart position')
    plt.plot(time, [x for x, xdot in positions], label='Position (m)')
    plt.plot(time, [xdot for x, xdot in positions], label='Accelaration(m/s^2)')
    plt.xlabel('Time (s)')
    plt.grid()
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.title('Cart angle')
    plt.plot(time, [tdot for t, tdot in angles], label='Angle variation (deg/s)')
    plt.plot(time, [t for t, tdot in angles], label='Angle (deg)')
    plt.plot([0, time[-1]], [180, 180], color='gray', linestyle='--')
    plt.xlabel('Time (s)')
    plt.grid()
    plt.legend()

    plt.show()


class BangBangCartPoleEnvironment(Environment):
    @save_args
    def __init__(self, force_factor=5, initial_theta=0.0001, max_offset=3, max_angle=0.25):
        self.norm_x = Normalizer(-max_offset, max_offset)
        self.norm_xdot = Normalizer(-10, 10)
        self.norm_theta = Normalizer(-max_angle, max_angle)
        self.norm_thetadot = Normalizer(-10, 10)
        self.reset()
    
    def reset(self):
        self.pendulum = PendulumDynamics(0, 0, self.initial_theta, 0)

    @property
    def state_size(self):
        return 4

    @property
    def action_size(self):
        return 3
    
    @property
    def state(self):
        return (
            self.norm_x(self.pendulum.x),
            self.norm_xdot(self.pendulum.xdot),
            self.norm_theta(self.pendulum.theta),
            self.norm_thetadot(self.pendulum.thetadot),
        )

    def _get_reward(self):
        if self.is_terminal:
            return -10
        else:
            return -0.005 * abs(self.pendulum.theta) - 0.001 * abs(self.pendulum.x)

    @property
    def is_terminal(self):
        return (
            not self.norm_x.is_inside(self.pendulum.x) or 
            not self.norm_xdot.is_inside(self.pendulum.xdot) or 
            not self.norm_theta.is_inside(self.pendulum.theta) or 
            not self.norm_thetadot.is_inside(self.pendulum.thetadot)
        )

    def apply_action(self, action_distribution):
        if self.is_terminal:
            raise RuntimeError('environment is in terminal state; cannot proceed further')

        act = np.random.choice([-1, 0, 1], p=action_distribution)
        self.pendulum.step_simulate(self.force_factor * act)
        return self.is_terminal, self.state, self._get_reward()


def drop_test():
    print('simple drop test, do the plots look realistic?')

    cart = PendulumDynamics(0, 0, 0.05, 0)

    time = [x * cart.dt for x in range(250)]
    positions, angles = [], []
    for i in time:
        positions.append((cart.x, cart.xdot))
        angles.append((to_deg(cart.theta), to_deg(cart.thetadot)))

        cart.step_simulate(0)

    plot(time, positions, angles)


def balance_test():
    print('trying to balance the thing...')

    cart = PendulumDynamics(0, 0, 0.05, 0)

    time = [x * cart.dt for x in range(250)]
    positions, angles = [], []
    for i in time:
        positions.append((cart.x, cart.xdot))
        angles.append((to_deg(cart.theta), to_deg(cart.thetadot)))

        force = cart.theta * -70
        cart.step_simulate(force)

    plot(time, positions, angles)


if __name__ == '__main__':
    drop_test()
    balance_test()
