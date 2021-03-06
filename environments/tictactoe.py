import numpy as np
from learning_task import Environment, LifeGoal


class TicTacToe:
    def __init__(self):
        self.state = '---------'

    def valid_move(self, move, state=None):
        state = state or self.state
        return 0 <= move < len(state) and state[move] == '-'

    def try_move(self, move, player, state=None):
        """
        Returns the resulting state after the player has performed the given move
        (does not alter game state).
        """
        state = state or self.state
        assert self.valid_move(move, state)
        return state[0:move] + player + state[move + 1:]

    def do_move(self, player, move):
        """
        Actually perform the given move by the player.
        """
        assert self.valid_move(move)
        self.state = self.try_move(move, player)

    def at(self, x, y):
        return self.state[3 * y + x]

    def moves(self, player, state=None):
        """
        Returns a list containing all the valid moves in the given or current state.
        Each element is a tuple (move, resulting state).
        """
        state = state or self.state
        blanks = [i for i, c in enumerate(state) if c == '-']
        return [(b, self.try_move(b, player, state)) for b in blanks]

    def winner(self, state=None):
        """
        Returns the winner in the given or current state, draw if the game has ended
        with a draw and None if there is no winner (game still in progress).
        """
        state = state or self.state
        if self.would_win('x', state):
            return 'x'
        elif self.would_win('o', state):
            return 'o'
        elif not '-' in state:
            return 'draw'
        else:
            return None

    def would_lose(self, player, state=None):
        """
        Returns wheter the given state is a losing state for the specified player.
        """
        state = state or self.state
        player = ['x', 'o'][player == 'x']
        return self.would_win(player, state)

    def would_win(self, player, state=None):
        """
        Returns wheter the given state is a winning state for the specified player.
        """
        state = state or self.state

        # check for horizontal win: 'xxx------', '---xxx---' and '------xxx'
        if (state[0] == state[1] == state[2] == player or
            state[3] == state[4] == state[5] == player or
            state[6] == state[7] == state[8] == player):
            return True

        # check for vertical win: 'x--x--x--', '-x--x--x-' and '--x--x--x'
        if (state[0] == state[3] == state[6] == player or
            state[1] == state[4] == state[7] == player or
            state[2] == state[5] == state[8] == player):
            return True

        # check for diagonal win: 'x---x---x' and '--x-x-x--'
        if (state[0] == state[4] == state[8] == player or
            state[2] == state[4] == state[6] == player):
            return True


class TicTacToeEnvironment(Environment):
    def __init__(self):
        self.onehot_encode = {
            '-': (0, 0, 1),
            'x': (0, 1, 0),
            'o': (1, 0, 0),
        }
        self.onehot_decode = {e: s for s, e in self.onehot_encode.items()}

        self.reset(None)

    def reset(self, agent_ids):
        self.ttt = TicTacToe()

        if agent_ids:
            sids = sorted(agent_ids)
            self.agent_map = {sids[0]: 'x', sids[1]: 'o'}

    @property
    def number_of_agents(self):
        return 2

    @property
    def state_size(self):
        return len(self.ttt.state) * 3

    @property
    def action_size(self):
        return len(self.ttt.state)

    def encode_state(self, state):
        return tuple([h for s in state for h in self.onehot_encode[s]])

    def decode_state(self, state):
        return ''.join(
            self.onehot_decode[state[i:i + 3]]
            for i in range(0, len(state), 3)
        )

    @property
    def state(self):
        return self.encode_state(self.ttt.state)

    @property
    def is_terminal(self):
        return self.ttt.winner is not None

    def apply_action(self, agent, action):
        try:
            self.ttt.do_move(self.agent_map[agent], np.argmax(action))
        except AssertionError:
            terminal = True
        else:
            terminal = self.ttt.winner() is not None

        return terminal, self.state


class TicTacToeWinGoal(LifeGoal):
    def get_reward_for_agent(self, agent, prev_state, action, state):
        prev_state_dec = self.environment.decode_state(prev_state)
        prev_action_dec = np.argmax(action)

        if self.environment.ttt.valid_move(prev_action_dec, prev_state_dec):
            return self._get_reward(agent)
        else:
            return None

    def get_end_reward_for_agent(self, agent, final_state, terminal):
        if self.environment.ttt.winner() is not None:
            return self._get_reward(agent)
        else:
            return None

    def _get_reward(self, agent):
        winner = {
            'x': 0,
            'o': 1,
            None: -1,
            'draw': -1,
        }[self.environment.ttt.winner()]

        if winner < 0:
            return 0
        elif winner == agent:
            return 1
        else:
            return -1
