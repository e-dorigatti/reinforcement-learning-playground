from learning_task import Agent
import numpy as np
from utils import save_args


class TableQLearningAgent:

    @save_args
    def __init__(self, exploration_coeff, lrate_coeff, discount_factor):
        self.Q, self.visited = {}, {}

    @save_args
    def build(self, state_size, action_size):
        pass

    def after_my_action(self, state, action, reward, next_state, terminal):
        act = np.argmax(action)
        if reward is None and act in self.Q[state]:
            self.Q[state].pop(act)

    def after_other_action(self, state, action, reward, next_state, terminal):
        act = np.argmax(action)
        if reward is None and state in self.Q and act in self.Q[state]:
            self.Q[state].pop(act)

    def episode_start(self, episode_number, state):
        self._match = []

    def _init_q(self, state):
        """ Lazy initialization of the tables needed to learning. """
        if state not in self.Q:
            initial = {action: np.random.random() for action in range(self.action_size)}

            self.Q[state] = initial
            self.visited[state] = dict(initial)

    def _update_q(self, state, action, value):
        self._init_q(state)
        self.visited[state][action] += 1
        self.Q[state][action] += value

    def exploration_function(self, state, action):
        """
        Alter the quality of a state-action pair to promote
        exploration of rarely visited state-action pairs.
        """
        quality = self.Q[state][action]
        visited_count = self.visited[state][action]

        return (quality + np.exp(-self.exploration_coeff * visited_count))

    def get_action(self, state):
        self._init_q(state)

        best_action = max(self.Q[state],
                          key=lambda a: self.exploration_function(state, a))

        self._match.append((state, best_action))

        onehot = [0] * self.action_size
        onehot[best_action] = 1
        return tuple(onehot)

    def episode_end(self, episode_number, end_state, terminal, reward):
        if reward is None:
            return

        seen_states = list(zip(self._match, self._match[1:] + [(None, None)]))
        for (state, action), (next_state, _) in reversed(seen_states):
            lrate = np.exp(-self.lrate_coeff * episode_number)
            estimated_optimal_future = (max(self.Q[next_state].values())
                                        if next_state else 0.0)
            learned = reward + self.discount_factor * estimated_optimal_future
            self._update_q(state, action, lrate * (learned - self.Q[state][action]))

            reward = 0  # reward is given only when winning or losing
