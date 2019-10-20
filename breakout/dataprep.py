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




def signal(n):
    if n == 0:
        return 0
    return 1

def process(ob):
    #ob = Image.fromarray(ob.astype('uint8'), 'RGB')
    #resize = transforms.Resize(64, 64)
    #ob = resize(ob)
    ob = torch.FloatTensor(ob.reshape(1, 3, 210, 160)).cuda()
    return ob


def to_grayscale(img):
    return np.mean(img, axis=2).astype(np.uint8)

def downsample(img):
    return img[::2, ::2]

def preprocess(img):
    img = downsample(img[26:, 10:-10, :])
    img = torch.FloatTensor(img.reshape(1,3,92,70)).cuda()
    return img

def policy_update(optimizer, log_prob, reward):
    score = reward*log_prob
    (score).backward()
    optimizer.zero_grad()
    optimizer.step()
    




env = gym.make("Breakout-v0")

print('env-made')
policy = PolicyNet(3,3).cuda() 
print('policy-ready')

optimizer = torch.optim.Adam(policy.parameters(), lr=0.003)
observation = env.reset()
running_reward = 0

for _ in range(200000):
    observation = preprocess(observation)

    output = policy(observation) # your agent here (this takes random actions)
    prob_dis = torch.distributions.Categorical(output)
    action = prob_dis.sample().item()
    #prob, action = torch.max(output, 1)
    action_signal = action + signal(action)*1
    env.step(1)
    
    observation, reward, done, info = env.step(action_signal)
    policy_update(optimizer, output[0, action], reward)
    running_reward += reward

    env.render()
    if done:
        print('******done*******')
        print(running_reward)
        running_reward = 0
        observation = env.reset()
env.close()



