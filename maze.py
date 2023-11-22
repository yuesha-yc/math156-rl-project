import numpy as np
from collections import namedtuple
import random
import time

"""
- states
    - 0 means free
    - -1 mean not traversable
    - 1 means goal
"""

Agent = namedtuple('Agent', ['i', 'j'])

class Agent:
    def __init__(self, i=0, j=0):
        self.i = i
        self.j = j

    @property
    def loc(self):
        return(self.i, self.j)
    
    def vmove(self, direction):
        direction = 1 if direction > 0 else -1
        return Agent(self.i + direction, self.j)
    
    def hmove(self, direction):
        direction = 1 if direction > 0 else -1
        return Agent(self.i, self.j + direction)

    def __repr__(self):
        return str(self.loc)

class QLearning:
    def __init__(self, num_states, num_actions, lr=0.1, discount_factor = 1.0):
        self.q = np.zeros((num_states, num_actions))
        self.a = lr
        self.g = discount_factor

    def update(self, st, at, rt, st1):
        q = self.q
        a = self.a
        g = self.g
        q[st, at] = (1 - a)*q[st, at] + a * (rt + g * np.max(q[st1]))



class Maze:
    def __init__(self, rows=4, columns=4):
        self.env = np.zeros((4, 4))
        self.mousy = Agent(0, 0)
        self.q = np.zeros((rows*columns, 4))

    def state_for_agent(self, a):
        nr, nc = self.env.shape
        return a.i * nc + a.j
    
    def in_bounds(self, i, j):
        nr, nc = self.env.shape
        return i >= 0 and i < nr and j >= 0 and j < nc
        
    def agent_in_bounds(self, a):
        return self.in_bounds(a.i, a.j)
    
    def agent_dient(self, a):
        return not self.env[a.i, a.j] == -1
    
    def is_valid_new_agent(self, a):
        return self.agent_in_bounds(a) and self.agent_dient(a)
    
    @property
    def all_acitons(self):
        a = self.mousy
        return [
            a.vmove(1),
            a.vmove(-1),
            a.hmove(1),
            a.hmove(-1)
        ]

    def compute_possible_moves(self):
        moves = self.all_acitons
        a = self.mousy
        return [(m, ii) for ii, m in enumerate(moves) if self.is_valid_new_agent(m)]

    def do_a_move(self, a):
        assert self.is_valid_new_agent(a), "Mousy can't go there"
        self.mousy = a
        return 10 if self.has_won() else -0.1

    def has_won(self):
        a = self.mousy
        return self.env[a.i, a.j] == 1

    def visualize(self):
        assert self.agent_in_bounds(self.mousy), "Mousy is out of bounds"
        e = self.env.copy()
        m = self.mousy
        e[m.i, m.j] = 6
        print(e)

def make_test_maze():
    m = Maze()
    e = m.env
    e[3, 3] = 1
    e[0, 1:3] = -1
    e[1, 2:] = -1
    e[3, 0:2] = -1
    return m

def main():
    
    m = make_test_maze()
    q = QLearning(16, 4)

    for i in range(100):
        final_score = 0
        m = make_test_maze()
        while not m.has_won():
            moves = m.compute_possible_moves()
            random.shuffle(moves)
            move, move_idx = moves[0]

            at = move_idx
            st = m.state_for_agent(m.mousy)

            score = m.do_a_move(move)
            final_score += score
            rt = score

            st1 = m.state_for_agent(m.mousy)

            q.update(st, at, rt, st1)

        # m.visualize()
        print(f"finished episode with final score of {final_score}")

    m = make_test_maze()
    m.visualize()
    final_score = 0
    while not m.has_won():
        time.sleep(0.1)
        s = m.state_for_agent(m.mousy)
        a_idx = np.argmax(q.q[s])
        final_score += m.do_a_move(m.all_acitons[a_idx])
        m.visualize()

    m.visualize()
    print(final_score)
if __name__ == '__main__':
    main()
    