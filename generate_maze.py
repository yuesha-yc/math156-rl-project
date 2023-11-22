import numpy as np
import random
import matplotlib.pyplot as plt 

# I want to generate maze
# 1. Start from left top corner
# 2. End at right bottom corner

# 3. Have a path from start to end
# 4. Have a path from start to end that is not straight
# 5. Have a path from start to end that is not straight and has a loop
# 6. Have a path from start to end that is not straight and has a loop and has a branch

# Finally, randomly place obstacles that is not on the path


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
    maze = np.zeros((N,N))
    maze[0,0] = -1 # start
    maze[N-1,N-1] = 2 # end

    # valid path: 1
    # invalid path 0
    # obstacles: 3

    # generate a path from start to end
    for i in range(num_paths):
        current = (0,0)
        while current != (N-1,N-1):
            next = get_valid_next(current,N)
            if next == (N-1,N-1):
                break
            maze[next] = 1
            current = next

    return maze


def add_obstacles(maze):
    N = maze.shape[0]
    # add obstacles to where there is no path
    for i in range(N):
        for j in range(N):
            if maze[i,j] == 0:
                maze[i,j] = random.choice([0,3])
    for i in range(N):
        for j in range(N):
            if maze[i,j] == 0:
                maze[i,j] = 1
    return maze


def visualize_maze(maze):
    plt.imshow(maze)
    plt.show()
    # save to file
    plt.savefig("maze.png")


if __name__ == "__main__":
    N = 10
    maze = gen_polygonal_path_maze(N,1)
    for i in range(10):
        new_maze = maze.copy()
        new_maze = add_obstacles(new_maze)
        plt.imshow(new_maze)
        # save to file
        plt.savefig(f"maze_{i}.png")
        plt.show()
        plt.close()
        # visualize_maze(maze)