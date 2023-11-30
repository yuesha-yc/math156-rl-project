import matplotlib.pyplot as plt

def visualize_path(maze, path):
    nr, nc = maze.env.shape  
    plt.figure(figsize=(nc, nr)) # size depend on matrix size
    plt.imshow(maze.env, cmap='gray') # grey tone
    plt.text(0, 0, 'start', va='center', ha='center', color='red', fontsize=20) # label "start"
    for position in path:
        plt.text(position[1], position[0], '->', va='center', color='green', fontsize=25) # label arrows

    plt.text(nc - 1, nr - 1, 'end', va='center', ha='center', color='red', fontsize=20) # label "end"
    plt.xticks([]), plt.yticks([])
    plt.show()
