import os
from agents.tableqlearning import TableQLearningAgent
from environments.tictactoe import TicTacToeEnvironment, TicTacToeWinGoal
from learning_task import LearningTask


class MonitorVictories(LearningTask):
    def __init__(self, debug_every, *args, **kwargs):
        super(MonitorVictories, self).__init__(*args, **kwargs)

        self.debug_every = debug_every
        self.reset()

    def reset(self):
        self.win_count = {'x': 0, 'o': 0, 'draw': 0, 'inv': 0}

    def print_debug(self):
        print('Episode {:<6} - WX: {x:<6} XO: {o:<6} DS: {draw:<6} INV: {inv:<6}'.format(
            self.episode_number, **self.win_count
        ))

    def do_episode(self):
        if self.episode_number % self.debug_every == 0:
            self.print_debug()
            self.reset()

        super(MonitorVictories, self).do_episode()

        winner = self.environment.ttt.winner() or 'inv'
        self.win_count[winner] += 1


def main():
    agents = [
        TableQLearningAgent(
            discount_factor=0.9,
            exploration_coeff=0.01,
            lrate_coeff=0.00001,
        ) for _ in range(2)
    ]

    environment = TicTacToeEnvironment()

    task = MonitorVictories(
        1000,
        environment,
        TicTacToeWinGoal(environment),
        agents,
        episode_length=-1,
        randomize_agents_order=True
    )

    task.run_episodes(10000000)


if __name__ == '__main__':
    main()
