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
        size = self.env.shape[0]
        if self.has_won():
            return 100 + size * 2 - 3   # reward for terminal
        elif self.agent_trape(a):
            return -10 # penalty for trap
        else:
            return -1 # penalty for transition

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

def train(q, training_mazes, testing_mazes, N_epoch = 100, directory="", visualize=False):

    testing_scores = []
    training_scores = []
    recorded_epochs = []

    for i in range(N_epoch + 1):
        final_score = 0
        # randomly choose a maze in training mazes
        new_maze = random.choice(training_mazes)
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

        if i % 1 == 0:
            recorded_epochs.append(i)

            # test on training mazes
            all_train_scores = []
            for j, training_maze in enumerate(training_mazes):
                train_score = test(q, training_maze, directory, f"epoch_{i}_train_{j}", visualize and j == 0)
                all_train_scores.append(train_score)
            
            # test on testing mazes
            all_test_scores = []
            for j, testing_maze in enumerate(testing_mazes):
                # print(f"Test maze {j}")
                test_score = test(q, testing_maze, directory, f"epoch_{i}_test_{j}", visualize and j == 0)
                all_test_scores.append(test_score)
            
            average_train_score = np.mean(all_train_scores)
            average_test_score = np.mean(all_test_scores)

            print(f"Epoch: {i}, Average train score: {average_train_score}, Average test score: {average_test_score}")

            training_scores.append(average_train_score)
            testing_scores.append(average_test_score)

    return recorded_epochs, training_scores, testing_scores


def test(q, test_maze, directory, appendix, visualize):
    maze_size = test_maze.shape[0]
    final_score = 0
    m = Maze(test_maze)
    all_states = [] # path taken in form of (i,j)
    all_actions = [] # arrows (0: up, 1: down, 2: right, 3: left)
    max_steps = maze_size * 2 - 2
    while not m.has_won() and max_steps > 0:
        max_steps -= 1
        # time.sleep(0.1)
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
    
    if visualize:
        visualize_path(m, all_states, f"{directory}/maze_path_{appendix}.png")
    return final_score


def train_single_maze(maze_size, path_num, obs_ratio):
    basic_path_maze = gen_polygonal_path_maze(maze_size, path_num)
    obstacle_maze = add_random_obstacles(basic_path_maze, obs_ratio)
    q = QLearning(100, 4)
    train(q, [obstacle_maze], N_epoch=100)


def train_generalized_maze(maze_size, path_num, obs_ratio, train_size = 10, test_size = 3, num_epochs = 100, visualize=False):
    directory = f"N={maze_size}_P={path_num}_R={obs_ratio}_trainsize={train_size}_testsize={test_size}_epochs={num_epochs}"

    # create directory if not exist
    if not os.path.exists(directory):
        os.makedirs(directory)

    basic_path_maze = gen_polygonal_path_maze(maze_size, path_num)

    maze_name = directory + "/maze_initial.png"
    visualize_maze(Maze(basic_path_maze), maze_name)

    training_mazes = []
    for i in range(train_size):
        training_mazes.append(add_random_obstacles(basic_path_maze, obs_ratio))

    testing_mazes = []
    for i in range(test_size):
        obstacle_maze_test = add_random_obstacles(basic_path_maze, obs_ratio)
        testing_mazes.append(obstacle_maze_test)

    q = QLearning(maze_size**2, 4)

    recorded_epochs, training_scores, testing_scores = train(q, training_mazes, testing_mazes, N_epoch=num_epochs, directory=directory, visualize=visualize)

    # plot values in testing scores
    plt.plot(recorded_epochs, training_scores)
    plt.xlabel("Number of epochs")
    plt.ylabel("Training Score")
    plt.savefig(f"{directory}/training_scores.png")
    plt.close()

    plt.plot(recorded_epochs, testing_scores)
    plt.xlabel("Number of epochs")
    plt.ylabel("Testing Score")
    plt.savefig(f"{directory}/testing_scores.png")
    plt.close()

    # save them as npy files
    np.save(f"{directory}/recorded_epochs.npy", recorded_epochs)
    np.save(f"{directory}/training_scores.npy", training_scores)
    np.save(f"{directory}/testing_scores.npy", testing_scores)

if __name__ == '__main__':

    # train_generalized_maze(10, 1, 0.5, train_size=5, test_size=5, num_epochs=500, visualize=True)

    # Ns = [10, 15, 20]
    # Ps = [1, 2, 3]
    # Rs = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    # training_size = 50
    # testing_size = 20
    # num_epochs = 500

    # Ns = [20]
    # Ps = [1, 2, 3, 4, 5]
    # Rs = [0.3,0.4,0.5,0.6,0.7]
    # training_size = 20
    # testing_size = 10
    # num_epochs = 300

    # Ns = [5,10,15,20,25,30,35,40,45,50]
    # Ps = [3]
    # Rs = [0.5]
    # training_size = 10
    # testing_size = 10
    # num_epochs = 1000

    # Ns = [40]
    # Ps = [3]
    # Rs = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
    # training_size = 10
    # testing_size = 10
    # num_epochs = 1000

    # for N in Ns:
    #     for P in Ps:
    #         for R in Rs:
    #             train_generalized_maze(N, P, R, training_size, testing_size, num_epochs)


    # load the npy files and plot them, with legend being different values of N
    # training_size = 50
    # testing_size = 20
    # num_epochs = 500
    # Ns = [10, 15, 20]
    # Ps = [1, 2, 3]
    # Rs = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    # for N in [10, 15, 20]:
    #     for P in [2]:
    #         for R in [0.4]:
    #             directory = f"N={N}_P={P}_R={R}_trainsize={training_size}_testsize={testing_size}_epochs={num_epochs}"
    #             recorded_epochs = np.load(f"{directory}/recorded_epochs.npy")
    #             training_scores = np.load(f"{directory}/training_scores.npy")
    #             testing_scores = np.load(f"{directory}/testing_scores.npy")
    #             plt.plot(recorded_epochs, testing_scores, label=f"N={N}_P={P}_R={R}")
    # plt.xlabel("Number of epochs")
    # plt.ylabel("Testing Score")
    # plt.legend()
    # plt.savefig(f"testing_scores_N_trainsize={training_size}_testsize={testing_size}_epochs={num_epochs}png")
    # plt.close()

    # Ns = [40]
    # Ps = [3]
    # # Rs = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
    # Rs = [0.1,0.3,0.9]
    # training_size = 10
    # testing_size = 10
    # num_epochs = 1000
    # for N in Ns:
    #     for P in Ps:
    #         for R in Rs:
    #             directory = f"N={N}_P={P}_R={R}_trainsize={training_size}_testsize={testing_size}_epochs={num_epochs}"
    #             recorded_epochs = np.load(f"{directory}/recorded_epochs.npy")
    #             training_scores = np.load(f"{directory}/training_scores.npy")
    #             testing_scores = np.load(f"{directory}/testing_scores.npy")
    #             plt.plot(recorded_epochs, testing_scores, label=f"N={N}_P={P}_R={R}")
    # plt.xlabel("Number of epochs")
    # plt.ylabel("Testing Score")
    # plt.ylim(-110,110)
    # plt.legend()
    # plt.savefig(f"testing_scores_Rs_trainsize={training_size}_testsize={testing_size}_epochs={num_epochs}png")
    # plt.close()
