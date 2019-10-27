import torch
import numpy as np
import cv2
import torch.optim as optim


class PolicyOptimizer():
    """This is going to update the policy gradients by keeping track
        of all the rewards and gradients of the log probability in each timestep
        and do a full backward pass at the end"""
    def __init__(self, policy_net):
        self.policy_net = policy_net
        self.probs = []
        self.rewards = []
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001)
        self.gamma = 0.99
    
    def cache_step(self, prob, reward):
        """Called at each time step to save the reward and the probability of the action taken"""

        self.rewards.append(reward)
        self.probs.append(prob)


    def discounted_reward(self, reward_list):
        """Discounts all the rewards of a trajectory into one discounted reward"""
        pw = 1
        discounted_reward = 0
        for r in reversed(reward_list):
            discounted_reward += r*(self.gamma**pw)
            pw += 1
        return discounted_reward

        
    def reinforce(self):
        gt = self.discounted_reward(self.rewards)
        for prob in self.probs:
            score = (-torch.log(prob)*gt)
            score.backward()

            self.optimizer.step()
            self.optimizer.zero_grad()
        self.probs = []
        self.rewards = []


        

        


def signal(n):
    """To only output the actions in [0, 2, 3"""
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
    """ Preprocess the state image
        cutting of the points bar on the top
        and the side gray bars"""
    img = downsample(img[26:, 10:-10, :])
    img = to_grayscale(img)
    img = cv2.resize(img, (84, 84))
    img = torch.FloatTensor(img.reshape(1,1,84,84)).cuda()
    return img

def policy_update(optimizer, log_prob, reward):
    """ At each step multiplying the reward and the log prob of the action taken
        in that state and calling backwards on that"""
    score = reward*log_prob
    (score).backward()
    optimizer.zero_grad()
    optimizer.step()
 


