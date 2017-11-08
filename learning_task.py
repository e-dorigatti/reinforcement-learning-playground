import tensorflow as tf


class Environment:
    """ An environment with which a number of agents can interact """
    def __init__(self, number_of_agents):
        assert number_of_agents > 0
        self.number_of_agents = number_of_agents
        self.turn = 0

    def reset(self):
        raise NotImplementedError()

    @property
    def next_agent(self):
        """ returns the index of the next agent that should take an action """
        return self.turn % self.number_of_agents

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

    def reward_for_agent(self, agent):
        """ returns the reward for the given agent """
        raise NotImplementedError()

    def apply_action(self, action):
        """
        applies the specified action, which must be a list of the appropriate length
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

    def episode_end(self, episode_number, state, terminal):
        """ called after the end of each episode """
        pass


class LifeGoal:
    """ The goal that should be achieved by the agents in a given environment """

    def __init__(self, environment):
        self.environment = environment

    def get_reward_for_agent(self, agent, prev_state, action, state):
        """
        gets the reward that should be given to the given agent
        for transitioning from a state to a next state with the given action
        """
        raise NotImplementedError()


class LearningTask:
    def __init__(self, environment, goal, agents, episode_length):
        assert len(agents) == environment.number_of_agents

        self.goal = goal
        self.episode_length = episode_length
        self.environment = environment
        self.agents = agents
        self.episode_number = 0

        for a in self.agents:
            a.build(self.environment.state_size, self.environment.action_size)

    def do_episode(self):
        self.episode_number += 1
        self.environment.reset()

        state = self.environment.state
        for a in self.agents:
            a.episode_start(self.episode_number, self.environment.state)

        terminal, i = False, 0
        while not terminal and (self.episode_length <= 0 or i < self.episode_length):
            player = self.environment.next_agent
            action = self.agents[player].get_action(self.environment.state)
            assert len(action) == self.environment.action_size

            terminal, next_state = self.environment.apply_action(action)
            for j, agent in enumerate(self.agents):
                reward = self.goal.get_reward_for_agent(
                    j, state, action, next_state if not terminal else None
                )

                cb = agent.after_my_action if j == player else agent.after_other_action
                cb(state, action, reward, next_state, terminal)

            state = next_state
            i += 1

        for a in self.agents:
            a.episode_end(self.episode_number, state, terminal)

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
