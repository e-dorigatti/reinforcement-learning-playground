import os
from agents.dqn import QNetworkAgent
from environments.cartpole import BangBangCartPoleEnvironment, CartPoleBalanceGoal
from learning_task import TaskWithTensorflow, LearningTask


def main():
    for f in os.listdir('logs'):
        os.remove('logs/' + f)

    agent = QNetworkAgent(
        discount_factor=0.99,
        target_network_track=0.001,
        regularization_coeff=0.001,
        learning_rate=0.01,
        hidden_state_size=64,
        hidden_size=32,
        random_action_decay=0.001,
        replay_buffer_size=1000000,
        batch_size=32
    )

    environment = BangBangCartPoleEnvironment(
        force_factor=5,
        initial_theta=0.01,
        max_offset=3,
        max_angle=0.25
    )

    task = TaskWithTensorflow(
        LearningTask(
            environment,
            CartPoleBalanceGoal(environment),
            [agent],
            episode_length=500,
            randomize_agents_order=True,
        ),
        device_count={'GPU': 0}
    )

    task.run_episodes(100000)


if __name__ == '__main__':
    main()
