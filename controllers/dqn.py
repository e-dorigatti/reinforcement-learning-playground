from utils import make_weights, compute_next_layer, KMostRecent, save_args
import numpy as np
from learning_task import Controller
import tensorflow as tf


class QNetwork:

    @save_args
    def __init__(self, state_size, action_size, tau, regularization_coeff,
                 learning_rate, hidden_state_size, hidden_size):
        pass

    @property
    def session(self):
        return tf.get_default_session()

    def get_onehot_action(self, action):
        if action is not None:
            act = [0] * self.action_size
            act[action] = 1
            return act
        else:
            return [self.get_onehot_action(i) for i in range(self.action_size)]

    def get_action(self, state):
        q_vals = self.session.run(self.output, feed_dict={
            self.nnet_input_state: np.array([state] * 3),
            self.nnet_input_action: np.array(
                self.get_onehot_action(None), dtype=np.float32
            )
        })
        return self.get_onehot_action(np.argmax(q_vals)), q_vals

    def get_q_values_from_target(self, states):
        in_states = [s for s in states for _ in range(3)]
        in_actions = [oa for _ in states for oa in self.get_onehot_action(None)]

        return self.session.run(self.target_output, feed_dict={
            self.nnet_input_state: in_states,
            self.nnet_input_action: in_actions
        })
    
    def learn(self, states, actions, outputs):
        _, loss_value = self.session.run([self.optimizer, self.loss], feed_dict={
            self.nnet_input_state: states,
            self.nnet_input_action: actions,
            self.nnet_label: outputs,
        })
        return loss_value

    def update_target_network(self):
        self.session.run(self.update_target_network_op)

    def build(self):
        self._build_network()
        self._build_target_network()

    def _build_network(self):
        self.nnet_input_state = tf.placeholder(
            shape=[None, self.state_size], dtype=tf.float32, name='nnet_input_state'
        )
        self.nnet_input_action = tf.placeholder(
            shape=[None, self.action_size], dtype=tf.float32, name='nnet_input_action'
        )

        self.nnet_label = tf.placeholder(
            shape=[None, 1], dtype=tf.float32, name='nnet_label'
        )

        self.weights_1, self.bias_1 = make_weights(self.state_size, self.hidden_state_size)
        self.hidden_state_1 = compute_next_layer(self.nnet_input_state, self.weights_1, self.bias_1)

        self.weights_2, self.bias_2 = make_weights(self.hidden_state_size, self.hidden_size)
        self.hidden_state_2 = compute_next_layer(self.hidden_state_1, self.weights_2, self.bias_2)

        self.weights_3, self.bias_3 = make_weights(self.action_size, self.hidden_size)
        self.hidden_action = compute_next_layer(self.nnet_input_action, self.weights_3, self.bias_3)

        self.bias_5 = tf.Variable(tf.constant(0.1, shape=[self.hidden_size]))
        self.hidden_combined = tf.nn.relu(self.hidden_state_2 + self.hidden_action + self.bias_5)

        self.weights_4, self.bias_4 = make_weights(self.hidden_size, 1)
        self.output = compute_next_layer(self.hidden_combined, self.weights_4, self.bias_4, activation=None)

        self.network_params = [self.weights_1, self.bias_1, self.weights_2,
                               self.bias_2, self.weights_3, self.bias_3,
                               self.weights_4, self.bias_4, self.bias_5]

        self.squared_error = (self.nnet_label - self.output)**2
        self.loss = tf.reduce_mean(self.squared_error) + self.regularization_coeff * sum(
            tf.reduce_sum(p **2) for p in self.network_params
        )

        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

    def _build_target_network(self):
        self.target_network_params = [tf.Variable(var.initialized_value())
                                      for var in self.network_params]
        self.update_target_network_op = [
            target_var.assign(self.tau * var + (1 - self.tau) * target_var)
            for var, target_var in zip(self.network_params, self.target_network_params)
        ]

        (
            self.target_weights_1, self.target_bias_1, self.target_weights_2,
            self.target_bias_2, self.target_weights_3, self.target_bias_3,
            self.target_weights_4, self.target_bias_4, self.target_bias_5
        ) = self.target_network_params

        self.target_hidden_state_1 = compute_next_layer(self.nnet_input_state, self.target_weights_1, self.target_bias_1)
        self.target_hidden_state_2 = compute_next_layer(self.target_hidden_state_1, self.target_weights_2, self.target_bias_2)
        self.target_hidden_action = compute_next_layer(self.nnet_input_action, self.target_weights_3, self.target_bias_3)
        self.target_hidden_combined = tf.nn.relu(self.target_hidden_state_2 + self.target_hidden_action + self.target_bias_5)
        self.target_output = compute_next_layer(self.target_hidden_combined, self.target_weights_4, self.target_bias_4, activation=None)


class QNetworkController(Controller):
    @save_args
    def __init__(self, discount_factor, target_network_track, regularization_coeff,
                 learning_rate, hidden_state_size, hidden_size, random_action_decay,
                 replay_buffer_size, batch_size):
        self.replay_buffer = KMostRecent(replay_buffer_size)

    def build(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.network = QNetwork(state_size, action_size, self.target_network_track,
                                self.regularization_coeff, self.learning_rate,
                                self.hidden_state_size, self.hidden_size)
        self.network.build()
    
    def episode_start(self, episode_number, state):
        self.episode_number = episode_number
        self.losses, self.rewards, self.qs, self.states, self.actions = [], [], [], [], []

    def _random_action_probability(self):
        return np.exp(-self.episode_number * self.random_action_decay)

    def get_action(self,  state):
        if np.random.random() < self._random_action_probability():
            act = np.random.randint(self.action_size)
            return self.network.get_onehot_action(act)
        else:
            action, q_v = self.network.get_action(state)
            self.qs.append(np.max(q_v))
            return action

    def action_performed(self, state, action, reward, next_state, terminal):
        self.replay_buffer.add((state, action, reward, next_state, terminal))
        self.rewards.append(reward)
        self.states.append(state)
        self.actions.append(action)

        batch_replay = self.replay_buffer.random_sample(self.batch_size)
        (
            batch_states, batch_actions, batch_rewards,
            batch_next_states, batch_terminal
        ) = zip(*batch_replay)
    
        next_q = self.network.get_q_values_from_target(batch_next_states)        
        batch_inputs_state, batch_inputs_action, batch_outputs = [], [], []
        for i, (_, _, reward, _, end) in enumerate(batch_replay):
            consider_future = 0 if end else 1
            batch_outputs.append([reward + consider_future * self.discount_factor * np.max(
                [next_q[self.action_size * i + j][0] for j in range(self.action_size)]
            )])

        loss = self.network.learn(batch_states, batch_actions, batch_outputs)
        self.network.update_target_network()
        assert not np.isnan(loss)

        self.losses.append(loss)
        return loss

    def episode_end(self, episode_number, state, terminal):
        summary = tf.Summary()

        sv = {
            'steps': len(self.rewards),
            'rnd': self._random_action_probability(),
        }

        sv['avg_reward'], sv['avg_loss'], sv['avg_q'] = list(map(
            np.mean, [self.rewards, self.losses, self.qs]
        ))
        sv['sum_reward'], sv['sum_loss'] = list(map(
            np.sum, [self.rewards, self.losses]
        ))

        for k, v in sv.items():
            summary.value.add(tag=k, simple_value=v)

        writer = tf.summary.FileWriter('logs')
        writer.add_summary(summary, global_step=self.episode_number)
        writer.close()

        with open('./logs/last-episode.csv', 'w') as f:
            for i, (state, action) in enumerate(zip(self.states, self.actions)):
                f.write(str(i) + ';' + ';'.join('%f' % x for x in tuple(state) + tuple(action)) + '\n')

        print('Episode %d - LEN: %d\tRND: %.3f\tLOS: %.3f\tAR: %.3f\tSR: %.3f\tAQ: %.3f' % (
            self.episode_number, sv['steps'], sv['rnd'], sv['avg_loss'], sv['avg_reward'],
            sv['sum_reward'], sv['avg_q'],
        ))