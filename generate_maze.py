import numpy as np
import random

# I want to generate NxN maze
# 1. Start from left top corner
# 2. End at right bottom corner
# 3. Have P paths from start to end
# 4. Obstacles are randomly placed at regions not on paths, 
# with obstacle to valid ratio being R from 0 to 1

def get_valid_next(current,N):
    next_x = min(current[0]+1,N-1)
    next_y = min(current[1]+1,N-1)
    next_option_1 = (next_x,current[1])
    next_option_2 = (current[0],next_y)
    next = random.choice([next_option_1, next_option_2])
    return next


# Given N, generate a maze of size N x N
# return a numpy array of size N x N
def gen_polygonal_path_maze(N,num_paths):
    maze = np.ones((N,N))
    maze[0,0] = 0 # start
    maze[N-1,N-1] = 1 # end

    # valid path: 0
    # invalid path -1

    # generate a path from start to end
    for i in range(num_paths):
        current = (0,0)
        while current != (N-1,N-1):
            next = get_valid_next(current,N)
            if next == (N-1,N-1):
                break
            maze[next] = 0
            current = next

    return maze


def add_random_obstacles(maze, obstacle_ratio):
    N = maze.shape[0]
    # Create a copy of the maze to modify
    modified_maze = maze.copy()
    # add obstacles to where there is no path
    for i in range(N):
        for j in range(N):
            if modified_maze[i,j] == 1:
                if i==N-1 and j==N-1:
                    continue
                # modified_maze[i,j] = random.choice([0,-1])
                if random.random() > obstacle_ratio:
                    modified_maze[i,j] = 0
                else:
                    modified_maze[i,j] = -1
                
    return modified_maze