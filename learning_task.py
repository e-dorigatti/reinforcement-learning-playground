import tensorflow as tf


class Environment:
    def reset(self):
        raise NotImplementedError()        

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

    def apply_action(self, action_distribution):
        raise NotImplementedError()


class Controller:
    def build(self, state_size, action_size):
        pass
    
    def episode_start(self, episode_number, state):
        pass

    def get_action(self, state):
        raise NotImplementedError()
    
    def action_performed(self, state, action, reward, next_state, terminal):
        pass
    
    def episode_end(self, episode_number, state, terminal):
        pass


class LearningTask:
    def __init__(self, controller, environment, episode_length):
        self.episode_length = episode_length
        self.environment = environment
        self.controller = controller
        self.episode_number = 0
        self.controller.build(self.environment.state_size,
                              self.environment.action_size)

    def do_episode(self):
        self.episode_number += 1

        self.environment.reset()

        state = self.environment.state
        self.controller.episode_start(self.episode_number, self.environment.state)

        for i in range(self.episode_length):
            action = self.controller.get_action(self.environment.state)
            terminal, next_state, reward = self.environment.apply_action(action)
            self.controller.action_performed(state, action, reward, next_state, terminal)

            state = next_state
            if terminal:
                break

        self.controller.episode_end(self.episode_number, state, terminal)

    def run_episodes(self, count):
        for _ in range(count):
            self.do_episode()


class TensorflowLearningTask(LearningTask):
    def __init__(self, controller, environment, episode_length, **tf_config):
        self.session = None
        self.tf_config = tf_config

        super(TensorflowLearningTask, self).__init__(
            controller, environment, episode_length
        )

    def do_episode(self):
        init = False
        if self.session is None:
            config = tf.ConfigProto(**self.tf_config) if self.tf_config else None
            self.session = tf.Session(config=config)
            init = True

        with self.session.as_default():
            if init:
                tf.global_variables_initializer().run()
            super(TensorflowLearningTask, self).do_episode()

    def __del__(self):
        if self.session:
            self.session.close()
