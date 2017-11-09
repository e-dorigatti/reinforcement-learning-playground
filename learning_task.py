import tensorflow as tf
from utils import save_args
import random


class Environment:
    """ An environment with which a number of agents can interact """

    def reset(self, agent_ids):
        raise NotImplementedError()

    @property
    def number_of_agents(self):
        """ returns the number of agents that must participate in this environment """
        raise NotImplementedError

    @property
    def state_size(self):
        raise NotImplementedError()

    @property
    def action_size(self):
        raise NotImplementedError()

    @property
    def state(self):
        raise NotImplementedError()

    @property
    def is_terminal(self):
        raise NotImplementedError()

    def apply_action(self, agent, action):
        """
        applies the specified action which must be a list of the appropriate length
        the agent is included in the list of IDs passed to the reset method
        return a tuple is_terminal, next_state, reward
        """
        raise NotImplementedError()


class Agent:
    """ A single agent that interacts with an environment """
    def build(self, state_size, action_size):
        """ initializee the controller """
        pass

    def episode_start(self, episode_number, state):
        """ called before the start of every episode """
        pass

    def get_action(self, state):
        """
        return the action to perform in the given state
        must be a list of the appropriate length (see the build method)
        """
        raise NotImplementedError()

    def after_my_action(self, state, action, reward, next_state, terminal):
        """ called after an action decided by this agent has been performed """
        pass

    def after_other_action(self, state, action, reward, next_state, terminal):
        """ called after an action decided by another agent has been performed """
        pass

    def episode_end(self, episode_number, state, terminal, reward):
        """ called after the end of each episode """
        pass


class LifeGoal:
    """ The goal that should be achieved by the agents in a given environment """

    def __init__(self, environment):
        self.environment = environment

    def get_reward_for_agent(self, agent, prev_state, action, state):
        """
        gets the reward that should be given to the agent
        for transitioning from a state to a next state with the given action
        the reward should be a scalar, or None if the action is not allowed in that state
        """
        raise NotImplementedError()

    def get_end_reward_for_agent(self, agent, final_state, terminal):
        """
        gets the reward that should be given to the agent
        when an episode ends in the specified state (whether terminal or not)
        """
        raise NotImplementedError()


class LearningTask:

    @save_args
    def __init__(self, environment, goal, agents,
                 episode_length=0, randomize_agents_order=True):
        assert len(agents) == environment.number_of_agents
        self.episode_number = 0

        for a in self.agents:
            a.build(self.environment.state_size, self.environment.action_size)

    def do_episode(self):
        self.episode_number += 1
        self.environment.reset(list(map(id, self.agents)))
        state = self.environment.state

        if self.randomize_agents_order:
            random.shuffle(self.agents)

        for a in self.agents:
            a.episode_start(self.episode_number, self.environment.state)

        terminal, i = False, 0
        while not terminal and (self.episode_length <= 0 or i < self.episode_length):
            action_agent = self.agents[i % len(self.agents)]
            action = action_agent.get_action(self.environment.state)
            assert len(action) == self.environment.action_size

            terminal, next_state = self.environment.apply_action(
                id(action_agent), action
            )

            for j, notify_agent in enumerate(self.agents):
                reward = self.goal.get_reward_for_agent(
                    j, state, action, next_state if not terminal else None
                )

                if notify_agent == action_agent:
                    cb = notify_agent.after_my_action
                else:
                    cb = notify_agent.after_other_action

                cb(state, action, reward, next_state, terminal)

            state = next_state
            i += 1

        for i, agent in enumerate(self.agents):
            reward = self.goal.get_end_reward_for_agent(i, state, terminal)
            agent.episode_end(self.episode_number, state, terminal, reward)

    def run_episodes(self, count):
        for _ in range(count):
            self.do_episode()


class TaskWithTensorflow:
    def __init__(self, task, **tf_config):
        self.session = None
        self.tf_config = tf_config
        self.task = task

    def do_episode(self):
        init = False
        if self.session is None:
            config = tf.ConfigProto(**self.tf_config) if self.tf_config else None
            self.session = tf.Session(config=config)
            init = True

        with self.session.as_default():
            if init:
                tf.global_variables_initializer().run()
            self.task.do_episode()

    def run_episodes(self, count):
        for _ in range(count):
            self.do_episode()

    def __del__(self):
        if self.session:
            self.session.close()
