import numpy as np
from collections import namedtuple
import random
import time
from PIL import Image
import pdb
import matplotlib.pyplot as plt 
from visualize_path import visualize_maze, visualize_path
import math

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
        if not self.is_valid_new_agent(a):
            assert self.is_valid_new_agent(a), "Mousy can't go there"
            print(f"Moving to {a.loc}")

        self.mousy = a
        if self.has_won():
            return 100   # reward for terminal
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

   
from generate_maze import *

def train(q, new_maze, N_epoch = 100, appendix="", maze_name="maze_with_obstacles.png"):

    visualize_maze(Maze(new_maze), maze_name[:-4] + f"_with_obstacles.png")

    training_process_scores = []
    training_scores = []
    training_path_lengths = []
    recorded_epochs = []

    for i in range(N_epoch):
        final_score = 0
        m = Maze(new_maze)
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

        training_process_scores.append(final_score)

        if i % 10 == 1:
            print(f"Epoch {i-1}")
            test_final_score, test_path_length = test(q, new_maze, appendix=f"epoch_{i-1}")
            print(f"Test final score: {test_final_score}")
            print(f"Test path length: {test_path_length}")
            recorded_epochs.append(i-1)
            training_scores.append(test_final_score)
            training_path_lengths.append(test_path_length)

    # test on training maze
    # print(f"Test on training maze: {appendix}")
    # test(q, new_maze, appendix)

    return recorded_epochs, training_process_scores, training_scores, training_path_lengths


def test(q, test_maze, appendix=""):
    maze_size = test_maze.shape[0]
    final_score = 0
    m = Maze(test_maze)
    all_states = [] # path taken in form of (i,j)
    all_actions = [] # arrows (0: up, 1: down, 2: right, 3: left)
    max_steps = maze_size * 2 - 2
    while not m.has_won() and max_steps > 0:
        max_steps -= 1
        time.sleep(0.1)
        s = m.state_for_agent(m.mousy)
        all_states.append((s//maze_size, s%maze_size))
        possible_actions = m.compute_possible_moves()
        q_vals = q.q[s]
        for q_idx, q_val in enumerate(q_vals):
            if q_idx not in [a_idx for _, a_idx in possible_actions]:
                # print(f"q_idx {q_idx} not in possible actions {possible_actions}")
                # print(f"q_vals {q_vals}")
                # print(f"state {s}")
                q_vals[q_idx] = -math.inf
        a_idx = np.argmax(q.q[s])
        all_actions.append(a_idx)
        final_score += m.do_a_move(m.all_acitons[a_idx])
        
    # Print results
    print(f"Final Score: {final_score}")
    path_length = len(all_states)
    print(f"Path Length: {path_length}")
    # print(f"Path taken: {all_states}")
    # print(f"Actions taken: {all_actions}")
    # print("Finished Maze:")
    visualize_path(m, all_states, f"maze_finished_{appendix}.png")

    return final_score, path_length


def train_single_maze(maze_size, path_num):
    basic_path_maze = gen_polygonal_path_maze(maze_size, path_num)
    obstacle_maze = add_random_obstacles(basic_path_maze)
    q = QLearning(100, 4)
    train(q, obstacle_maze, N_epoch=100)


def train_generalized_maze(maze_size, path_num, train_size = 10, test_size = 3, num_epochs = 100):
    basic_path_maze = gen_polygonal_path_maze(maze_size, path_num)

    maze_name = f"maze_initial_N={maze_size}_P={path_num}.png"
    visualize_maze(Maze(basic_path_maze), maze_name)

    training_mazes = []
    for i in range(train_size):
        training_mazes.append(add_random_obstacles(basic_path_maze))

    testing_mazes = []
    for i in range(test_size):
        obstacle_maze_test = add_random_obstacles(basic_path_maze)
        testing_mazes.append(obstacle_maze_test)

    q = QLearning(maze_size**2, 4)
    for i, training_maze in enumerate(training_mazes):
        recorded_epochs, training_process_scores, training_scores, training_path_lengths = train(q, training_maze, N_epoch=num_epochs, appendix=f"train_{i}", maze_name=maze_name)
        # plot values in training process scores
        plt.plot(recorded_epochs, training_process_scores)
        plt.xlabel("Number of epochs")
        plt.ylabel("Score")
        plt.savefig(f"training_process_scores_{i}.png")
        plt.close()

        # plot values in testing scores
        plt.plot(recorded_epochs, training_scores)
        plt.xlabel("Number of epochs")
        plt.ylabel("Training Score")
        plt.savefig(f"training_scores_{i}.png")
        plt.close()

        # plot values in testing path lengths
        plt.plot(recorded_epochs, training_path_lengths)
        plt.xlabel("Number of epochs")
        plt.ylabel("Path Length")
        plt.savefig(f"training_path_lengths_{i}.png")
        plt.close()


    
    for i, testing_maze in enumerate(testing_mazes):
        print(f"Test maze {i}")
        test(q, testing_maze, appendix=f"test_{i}")


if __name__ == '__main__':
    train_generalized_maze(20, 1, train_size=1, test_size=3, num_epochs=300)