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

#optimizer = torch.optim.Adam(policy.parameters(), lr=0.003)
p_optimizer = PolicyOptimizer(policy) 
observation = env.reset()
ep_rewards = []
num_episodes = 50000
running_rewards = 0

for ep in range(1, num_episodes+1):

    while True:


        try:
            observation = preprocess(observation) # State image size -> 84 X 84 

            action, prob = policy(observation) # your agent here (output: [action, probability of action])
            env.render()
            
            observation, reward, done, info = env.step(action)
            running_rewards += reward
            p_optimizer.cache_step(prob, reward)

            if done:
                ep_rewards.append(running_rewards)

                print('EP '+str(ep))
                print('ep-reward-------'+str(running_rewards))
                print("avg-ep-reward----"+str(sum(ep_rewards)/len(ep_rewards)))
                print('\n')
                running_rewards = 0 
                p_optimizer.reinforce()
                observation = env.reset()
                break

        except KeyboardInterrupt:
            path_name = input("enter checkpoint name: ")
            if path_name == "":
                torch.save(policy.state_dict(), 'checkpoint1.pth')
            torch.save(policy.state_dict(), path_name)
            exit()
env.close()



