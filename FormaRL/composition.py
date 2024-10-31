# Specification is: reward should never be -1, reach goals 1 then 2.

# PPO Implementation from https://github.com/nikhilbarhate99/PPO-PyTorch

import os
import glob
import time
from datetime import datetime

import torch
import numpy as np

import gym
# import roboschool

from custom_env import Custom_env

from PPO import PPO

start = [9, 1]
# start = [1, 1]
goals = [[1, 1], [3, 4], [1, 9]]
# goals = [[1, 9]]

################################### Training ###################################
def test():

    ####### initialize environment hyperparameters ######
    env_name = "compose"

    has_continuous_action_space = False  # continuous action space; else discrete

    max_ep_len = 1000                   # max timesteps in one episode
    max_training_timesteps = int(210000)    # break training loop if timeteps > max_training_timesteps

    print_freq = max_ep_len * 10        # print avg reward in the interval (in num timesteps)
    log_freq = max_ep_len * 2           # log avg reward in the interval (in num timesteps)
    save_model_freq = int(1e5)          # save model frequency (in num timesteps)

    action_std = 0.6                    # starting std for action distribution (Multivariate Normal)
    action_std_decay_rate = 0.05        # linearly decay action_std (action_std = action_std - action_std_decay_rate)
    min_action_std = 0.1                # minimum action_std (stop decay after action_std <= min_action_std)
    action_std_decay_freq = int(2.5e5)  # action_std decay frequency (in num timesteps)
    #####################################################

    ## Note : print/log frequencies should be > than max_ep_len

    ################ PPO hyperparameters ################
    update_timestep = max_ep_len * 4      # update policy every n timesteps
    K_epochs = 80               # update policy for K epochs in one PPO update

    eps_clip = 0.2          # clip parameter for PPO
    gamma = 0.99            # discount factor

    lr_actor = 0.0003       # learning rate for actor network
    lr_critic = 0.001       # learning rate for critic network

    random_seed = 0         # set random seed if required (0 = no random seed)
    #####################################################

    # env = gym.make(env_name)
    env = Custom_env()

    max_ep_len = 10000

    # state space dimension
    # state_dim = env.observation_space.shape[0]
    state_dim = env.state_dim
    action_dim = env.action_dim

    ################# Composing ################

    # initialize a PPO agent
    ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std)


    # initialize a PPO agent
    ppo_agent.load("/home/swaminathan/git/spectrl_tool/PPO_preTrained/goal-1/PPO_goal-1_0_0.pth")


    # printing and logging variables
    print_running_reward = 0
    print_running_episodes = 0

    log_running_reward = 0
    log_running_episodes = 0

    time_step = 0
    i_episode = 0


    state = env.reset()
    current_ep_reward = 0

    predicate_num = 0

    for t in range(1, max_ep_len+1):

        # select action with policy
        action = ppo_agent.select_action(state)
        if predicate_num == 2:
            # Reverse the actions
            if action == 0 or action == 1:
                action = action + 2
            else:
                action = action - 2
            state, reward, done, _ = env.step(action)
        else:
            state, reward, done, _ = env.step(action)

        rewards = ppo_agent.buffer.rewards
        # saving reward and is_terminals
        ppo_agent.buffer.rewards.append(reward)
        ppo_agent.buffer.is_terminals.append(done)

        # print(state)
        if state == goals[0] and predicate_num == 0:
            print("Reached [1, 1] from [9, 1]")
            predicate_num = 1

            done = False

            rewards = ppo_agent.buffer.rewards
            dones = ppo_agent.buffer.is_terminals

            ppo_agent.load("/home/swaminathan/git/spectrl_tool/PPO_preTrained/goal-3-4/PPO_goal-3-4_0_0.pth")
        
        elif state == goals[1] and predicate_num == 1:
            print("Reached [3, 4] from [1, 1]")
            predicate_num = 2

            done = False

            rewards = ppo_agent.buffer.rewards
            dones = ppo_agent.buffer.is_terminals

            ppo_agent.load("/home/swaminathan/git/spectrl_tool/PPO_preTrained/goal-1-9-to-3-4/PPO_goal-1-9-to-3-4_0_1.pth")
            
        elif np.sum(np.array(state) - np.array(goals[2])) < 1 and predicate_num == 2:
            
            print("Reached [1, 9] from [3, 4]. (Reversed policy). Succesfully composed..")
            done = True

        time_step +=1
        current_ep_reward += reward

        # break; if the episode is over
        if done:
            break

        print_running_reward += current_ep_reward
        print_running_episodes += 1

        log_running_reward += current_ep_reward
        log_running_episodes += 1

        i_episode += 1

    # log_f.close()
    env.close()


if __name__ == '__main__':
    test()

