from __future__ import print_function
import numpy as np
from math import sin, cos, pi
from utils import save_args
import matplotlib.pyplot as plt
from learning_task import Environment, LifeGoal
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


class BaseCartPoleEnvironment(Environment):
    @save_args
    def __init__(self, force_factor=5, initial_theta=0.0001,
                 max_offset=3, max_angle=0.25):
        super(BaseCartPoleEnvironment, self).__init__(number_of_agents=1)

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
        raise NotImplementedError()

    @property
    def state(self):
        return (
            self.norm_x(self.pendulum.x),
            self.norm_xdot(self.pendulum.xdot),
            self.norm_theta(self.pendulum.theta),
            self.norm_thetadot(self.pendulum.thetadot),
        )

    def denormalize_state(self, state):
        x, xdot, theta, thetadot = state
        return (
            self.norm_x.denormalize(x),
            self.norm_xdot.denormalize(xdot),
            self.norm_theta.denormalize(theta),
            self.norm_thetadot.denormalize(thetadot),
        )

    @property
    def is_terminal(self):
        return (
            not self.norm_x.is_inside(self.pendulum.x) or
            not self.norm_xdot.is_inside(self.pendulum.xdot) or
            not self.norm_theta.is_inside(self.pendulum.theta) or
            not self.norm_thetadot.is_inside(self.pendulum.thetadot)
        )

    @staticmethod
    def _get_force(action):
        raise NotImplementedError()

    def apply_action(self, action):
        if self.is_terminal:
            raise RuntimeError('environment is in terminal state; cannot proceed further')

        act = self._get_force(action)
        self.pendulum.step_simulate(self.force_factor * act)
        return self.is_terminal, self.state


class BangBangCartPoleEnvironment(BaseCartPoleEnvironment):
    @property
    def action_size(self):
        return 3

    @staticmethod
    def _get_force(action):
        idx = np.argmax(action)
        return idx - 1


class ContinuousCartPoleEnvironment(BaseCartPoleEnvironment):
    @property
    def action_size(self):
        return 1

    @staticmethod
    def _get_force(action):
        return min(max(action[0], -1), 1)


class CartPoleBalanceGoal(LifeGoal):
    def get_reward_for_agent(self, agent, prev_state, action, state):
        if state is None:
            return -10
        else:
            x, _, theta, _ = self.environment.denormalize_state(state)
            return 0.005 * (1 - abs(theta)) + 0.001 * (1 - abs(x))
