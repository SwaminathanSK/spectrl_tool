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

class Custom_env():
    def __init__(self) -> None:
        self.env_name = "custom_world"
        self.state_dim = 2
        self.action_dim = 4
        self.current_state = None

    def reset(self):
        self.current_state = [9, 1]
        return self.current_state
    
    def step(self, action):
        
        action = (action == 0)*"R" + (action == 1)*"U" + (action == 2)*"L" + (action == 3)*"D"
        
        try:
            next = nextState[self.current_state[0]][self.current_state[1]][action]
            reward = actionReward[self.current_state[0]][self.current_state[1]][action]
            done = False
        except:
            next = self.current_state
            reward = 0
            done = False
        if reward >= 1:
            done = True
        # print(reward)
        return next, reward, done, None
    
    def close():
        pass

