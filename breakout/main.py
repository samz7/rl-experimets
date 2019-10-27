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
policy = PolicyNet(3,3).cuda() 
print('policy-ready')

#optimizer = torch.optim.Adam(policy.parameters(), lr=0.003)
p_optimizer = PolicyOptimizer(policy) 
observation = env.reset()
running_reward = 0
time_steps = 0

while True:

    try:
        observation = preprocess(observation) # State image size -> 84 X 84 
        lives = env.ale.lives()

        output = policy(observation) # your agent here (this takes random actions)
        prob_dis = torch.distributions.Categorical(output)
        action = prob_dis.sample().item()
        #prob, action = torch.max(output, 1)
        action_signal = action + signal(action)*1
        env.step(1)
        env.render()
        
        observation, reward, done, info = env.step(action_signal)
        time_steps += 1
        running_reward += reward
        #policy_update(optimizer, output[0, action], reward)
        p_optimizer.cache_step(output[0, action], reward)

        if done: 

            print("average_rewards----"+str(running_reward/time_steps))
            
            running_reward = 0
            p_optimizer.reinforce()
            observation = env.reset()

    except KeyboardInterrupt:
        path_name = input("enter checkpoint name: ")
        if path_name == "":
            torch.save(policy.state_dict(), 'checkpoint1.pth')
        torch.save(policy.state_dict(), path_name)
        exit()
env.close()



