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

optimizer = torch.optim.Adam(policy.parameters(), lr=0.003)
observation = env.reset()
running_reward = 0
episode = 1

while True:

    try:
        observation = preprocess(observation) # State image size 

        output = policy(observation) # your agent here (this takes random actions)
        prob_dis = torch.distributions.Categorical(output)
        action = prob_dis.sample().item()
        #prob, action = torch.max(output, 1)
        action_signal = action + signal(action)*1
        env.step(1)
        env.render()
        
        observation, reward, done, info = env.step(action_signal)
        running_reward += reward
        policy_update(optimizer, output[0, action], reward)

        if done:
            print('episode----'+str(episode))
            print("rewards----"+str(running_reward))
            episode +=1
            
            running_reward = 0
            observation = env.reset()
    except KeyboardInterrupt:
        path_name = input("enter checkpoint name: ")
        if path_name == "":
            torch.save(policy.state_dict(), 'checkpoint1.pth')
        torch.save(policy.state_dict(), path_name)
        exit()
env.close()



