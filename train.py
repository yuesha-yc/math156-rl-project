import numpy as np
from collections import namedtuple
import random
import time
from PIL import Image
import pdb
import matplotlib.pyplot as plt 


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
    def __init__(self, rows=100, columns=100,):
        self.env = np.zeros((rows, columns))
        self.mousy = Agent(0, 0)
        self.q = np.zeros((rows*columns, 4))
    
    def __init__(self, m):
        self.env = m
        self.mousy = Agent(0, 0)
        self.q = np.zeros((m.shape[0] * m.shape[1], 4))

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

    def agent_trape(self, a):
        return self.env[a.i, a.j] == -1
    
    def is_valid_new_agent(self, a):
        return self.agent_in_bounds(a)
    
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
        if self.has_won():
            return 10   # reward for terminal
        elif self.agent_trape(a):
            return -0.3 # penalty for trap
        else:
            return -0.1 # penalty for transition

    def has_won(self):
        a = self.mousy
        return self.env[a.i, a.j] == 1

    def visualize(self):
        assert self.agent_in_bounds(self.mousy), "Mousy is out of bounds"
        e = self.env.copy()
        m = self.mousy
        e[m.i, m.j] = 6
        print(e)
        return e

# def make_test_maze():
#     m = Maze()
#     e = m.env
#     e[-1, -1] = 1
#     e[0, 1:3] = -1
#     e[1, 2:] = -1
#     e[3, 0:2] = -1
#     return m

# def update_q(times, q):
    
#     for i in range(times):
#         final_score = 0
#         m = make_test_maze()
#         while not m.has_won():
#             moves = m.compute_possible_moves()
#             random.shuffle(moves)
#             move, move_idx = moves[0]

#             at = move_idx
#             st = m.state_for_agent(m.mousy)

#             score = m.do_a_move(move)
#             final_score += score
#             rt = score

#             st1 = m.state_for_agent(m.mousy)

#             q.update(st, at, rt, st1)
#     return q

# def use_q(q):
#     m = make_test_maze()
#     final_score = 0
#     while not m.has_won():
#         time.sleep(0.1)
#         s = m.state_for_agent(m.mousy)
#         a_idx = np.argmax(q.q[s])
#         final_score += m.do_a_move(m.all_acitons[a_idx])
#     return final_score


# def main():
#     q = QLearning(10000, 4)
#     for i in range(100, 1000):
#         q = update_q(i, q)
#         print(use_q(q))

   
from generate_maze import *

def train(q, new_maze, N_epoch = 100, appendix=""):
    for i in range(N_epoch):
        final_score = 0
        m = Maze(new_maze)
        # print(m.visualize())
        while not m.has_won():
            moves = m.compute_possible_moves()
            random.shuffle(moves)
            move, move_idx = moves[0]

            at = move_idx
            st = m.state_for_agent(m.mousy)

            score = m.do_a_move(move)
            # print(f"Score: {score}")
            final_score += score
            rt = score

            st1 = m.state_for_agent(m.mousy)

            q.update(st, at, rt, st1)
        # print(f"Final Score, epoch {i}: {final_score}")

    # test on training maze
    print("Test on training maze:")
    test(q, new_maze, appendix)


def test(q, test_maze, appendix=""):
    maze_size = test_maze.shape[0]
    final_score = 0
    m = Maze(test_maze)
    all_states = [] # path taken in form of (i,j)
    all_actions = [] # arrows (0: up, 1: down, 2: right, 3: left)
    while not m.has_won():
        time.sleep(0.1)
        s = m.state_for_agent(m.mousy)
        all_states.append((s//maze_size, s%maze_size))
        a_idx = np.argmax(q.q[s])
        all_actions.append(a_idx)
        final_score += m.do_a_move(m.all_acitons[a_idx])

    # Print results
    print(f"Final Score: {final_score}")
    print(f"Path taken: {all_states}")
    print(f"Actions taken: {all_actions}")
    
    # Visualize Maze
    print("Finished Maze:")
    finished_maze = m.visualize()
    for state in all_states:
        finished_maze[state] = 5
    plt.imshow(finished_maze)
    plt.savefig(f"maze_finished_{appendix}.png")
    # plt.show()
    plt.close()


def train_single_maze(maze_size, path_num):
    basic_path_maze = gen_polygonal_path_maze(maze_size, path_num)
    obstacle_maze = add_random_obstacles(basic_path_maze)
    print("Initial Maze:")
    Maze(obstacle_maze).visualize()
    q = QLearning(100, 4)
    train(q, obstacle_maze, N_epoch=100)


def train_generalized_maze(maze_size, path_num, train_size = 10, test_size = 3):
    basic_path_maze = gen_polygonal_path_maze(maze_size, path_num)

    plt.imshow(basic_path_maze)
    plt.savefig(f"maze_finished_initial.png")
    plt.close()

    training_mazes = []
    for i in range(train_size):
        training_mazes.append(add_random_obstacles(basic_path_maze))

    testing_mazes = []
    for i in range(test_size):
        obstacle_maze_test = add_random_obstacles(basic_path_maze)
        testing_mazes.append(obstacle_maze_test)

    q = QLearning(maze_size**2, 4)
    for i, training_maze in enumerate(training_mazes):
        train(q, training_maze, N_epoch=100, appendix=f"train_{i}")
    
    for i, testing_maze in enumerate(testing_mazes):
        print(f"Test maze {i}")
        test(q, testing_maze, appendix=f"test_{i}")


if __name__ == '__main__':
    train_generalized_maze(20, 2, train_size=10, test_size=10)