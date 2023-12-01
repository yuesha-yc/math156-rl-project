import matplotlib.pyplot as plt
import numpy as np

def visualize_path(maze, path):
    nr, nc = maze.env.shape  
    plt.figure(figsize=(nc, nr))  # size depends on matrix size

    # add grid
    for i in range(nr + 1):
        plt.axhline(y=i - 0.5, color='black', linestyle='-', linewidth=1)
    for j in range(nc + 1):
        plt.axvline(x=j - 0.5, color='black', linestyle='-', linewidth=1)

    plt.imshow(maze.env)  #, cmap='gray')  # grey tone
    plt.text(0, 0, 'start', va='center', ha='center', color='red', fontsize=15)  # start label
    plt.text(nc - 1, nr - 1, 'end', va='center', ha='center', color='red', fontsize=15)  # end label

    # label arrow directions
    for i in range(len(path)):
        position = path[i]
        direction = get_arrow_direction(i, path)
        plt.text(position[1], position[0], direction, va='center', color='white', fontsize=25) 
    
    plt.xticks([]), plt.yticks([])
    plt.show()

def get_arrow_direction(current_index, path):
    if current_index == len(path) - 1:
        return ''  # No arrow for the end label
    
    current_pos = np.array(path[current_index])
    next_pos = np.array(path[current_index + 1])

    direction = next_pos - current_pos
    
    if direction[0] == 1:
        return '↓'  # Move down
    elif direction[0] == -1:
        return '↑'  # Move up
    elif direction[1] == 1:
        return '→'  # Move right
    elif direction[1] == -1:
        return '←'  # Move left
    else:
        return ''
