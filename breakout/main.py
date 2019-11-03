import gym
from torch.utils.data import DataLoader
from policy import PolicyNet
import gym
import torch
import random
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
from utils import *


env = gym.make("Breakout-v0")

print('env-made')
policy = PolicyNet(3,4).cuda() 
print('policy-ready')

p_optimizer = PolicyOptimizer(policy)
observation = env.reset()
reward_count = 0 
total_rewards = 0
num_episodes = 50000
time_steps = 5000
running_rewards = 0

for ep in range(1, num_episodes+1):

    for t in range(time_steps):
        ep_rewards = []

        try:
            observation = preprocess(observation) # State image size -> 84 X 84 

            action, prob = policy(observation) # your agent here (output: [action, probability of action])
            
            observation, reward, done, info = env.step(action)
            env.render()

            ep_rewards.append(reward)
            running_rewards += reward
            p_optimizer.cache_step(prob, reward)

            if done:
                total_rewards += running_rewards
                reward_count += len(ep_rewards)

                print('EP '+str(ep))
                print('ep-reward-------'+str(running_rewards))
                print("avg-ep-reward----"+str((total_rewards)/(reward_count)))
                print('\n')
                running_rewards = 0 
                p_optimizer.reinforce()
                observation = env.reset()
                break
            if t == time_steps-1:
                ep_rewards.append(running(rewards))
                print('EP '+str(ep))
                print('ep-reward-------'+str(running_rewards))
                print("avg-ep-reward----"+str(sum(ep_rewards)/len(ep_rewards)))
                print('\n')
                running_rewards = 0 
                p_optimizer.reinforce()
                break


        except KeyboardInterrupt:
            path_name = input("enter checkpoint name: ")
            if path_name == "":
                torch.save(policy.state_dict(), 'checkpoint1.pth')
            torch.save(policy.state_dict(), path_name)
            exit()
env.close()



