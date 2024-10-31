import numpy as np
import matplotlib.pyplot as plt
from matplotlib.table import Table

WORLD_SIZE = 9
discount = 0.9

start = [9, 1]
goals = [[1, 9], [1, 1], [7, 8]]

# leftgap = [5, 3] # left gap
# rightgap = [6, 7] # right gap
# upgap = [0, 5] # up gap
# downgap = [7, 5] # down gap


world = np.array(
        [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
         [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
         [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
         [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
         [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
         [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
         [1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1],
         [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
         [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
         [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])

# world = np.zeros((WORLD_SIZE, WORLD_SIZE))

# left, up, right, down
actions = ['L', 'U', 'R', 'D']

actionProb = []
for i in range(0, WORLD_SIZE+2):
    actionProb.append([])
    for j in range(0, WORLD_SIZE+2):
      actionProb[i].append(dict({'L':0.25, 'U':0.25, 'R':0.25, 'D':0.25}))

nextState = []
actionReward = []
for i in range(0, WORLD_SIZE+2):
   nextState.append([])
   actionReward.append([])
   for j in range(0, WORLD_SIZE+2):
      next = {'R' : [i, j], 'L' : [i, j], 'D' : [i, j], 'U' : [i, j]}
      reward = {'R' : -1, 'L' : -1, 'D' : -1, 'U' : -1}

      nextState[i].append(next)
      actionReward[i].append(reward)

for i in range(1, WORLD_SIZE+1):
    nextState.append([])
    actionReward.append([])
    for j in range(1, WORLD_SIZE+1):
        next = {'R' : [i, j+1], 'L' : [i, j-1], 'D' : [i+1, j], 'U' : [i-1, j]}
        reward = {'R' : 0, 'L' : 0, 'D' : 0, 'U' : 0}

        if world[next['R'][0], next['R'][1]] == 1:
           reward['R'] = -1.0
        if world[next['L'][0], next['L'][1]] == 1:
           reward['L'] = -1.0
        if world[next['D'][0], next['D'][1]] == 1:
           reward['D'] = -1.0
        if world[next['U'][0], next['U'][1]] == 1:
           reward['U'] = -1.0
        
        for goal in goals:
            if [next['R'][0], next['R'][1]] == goal:
                reward['R'] = 1.0
            if [next['U'][0], next['U'][1]] == goal:
                reward['U'] = 1.0
            if [next['L'][0], next['L'][1]] == goal:
                reward['L'] = 1.0
            if [next['D'][0], next['D'][1]] == goal:
                reward['D'] = 1.0

        nextState[i][j] = next
        actionReward[i][j] = reward


def draw_image(image):
    fig, ax = plt.subplots()
    ax.set_axis_off()
    tb = Table(ax, bbox=[0,0,1,1])

    nrows, ncols = image.shape
    width, height = 1.0 / ncols, 1.0 / nrows

    # Add cells
    for (i,j), val in np.ndenumerate(image):
        # Index either the first or second item of bkg_colors based on
        # a checker board pattern
        idx = [j % 2, (j + 1) % 2][i % 2]
        color = 'white'

        tb.add_cell(i, j, width, height, text=val,
                    loc='center', facecolor=color)

    # Row Labels...
    for i, label in enumerate(range(len(image))):
        tb.add_cell(i, -1, width, height, text=label, loc='right',
                    edgecolor='none', facecolor='none')
    # Column Labels...
    for j, label in enumerate(range(len(image))):
        tb.add_cell(-1, j, width, height/2, text=label, loc='center',
                           edgecolor='none', facecolor='none')
    ax.add_table(tb)
    plt.show()

li = []

# optimal
world = np.zeros((WORLD_SIZE+2, WORLD_SIZE+2))
while True:
    # keep iteration until convergence
    newWorld = np.zeros((WORLD_SIZE+2, WORLD_SIZE+2))
    for i in range(0, WORLD_SIZE+2):
        for j in range(0, WORLD_SIZE+2):
          if (i, j) not in li:
            values = []
            for action in actions:
                newPosition = nextState[i][j][action]
                # value iteration
                values.append(actionReward[i][j][action] + discount * world[newPosition[0], newPosition[1]])
            newWorld[i][j] = np.max(values)
    if np.sum(np.abs(world - newWorld)) < 1e-4:
        print('Optimal Policy')
        draw_image(np.round(newWorld, decimals=2))
        break
    world = newWorld