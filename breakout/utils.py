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



    def mul_rewards(self):
        for idx in self.reward_index:
            t = [] 
            for p, r in zip(self.probs[:idx], self.r_gen(len(self.probs[:idx]))):
                t.append(-p*r)
            self.probs[:idx] = t
        



    def r_gen(self, n):
        pw = 1
        discounted_reward = []
        for i in range(n):
            discounted_reward.append(1+(self.gamma**pw))
            pw += 1
        return torch.tensor(list(reversed(discounted_reward)))



    def discounted_reward(self, rewards):

        """Discounts all the rewards of a trajectory"""
        discounted_rewards = []

        for t in range(len(rewards)):
            Gt = 0
            pw = 0
            for r in rewards[t:]:
                Gt = Gt + self.gamma**pw * r
                pw = pw + 1
            discounted_rewards.append(Gt)
        discounted_rewards = torch.tensor(discounted_rewards)

        return discounted_rewards

                                        

        
    def reinforce(self):
        self.optimizer.zero_grad()
        rewards = self.discounted_reward(self.rewards)
        #self.mul_rewards()
        scores = []

        for log_prob, r in zip(self.probs, rewards):
            scores.append(-(log_prob*r))

        expectation = torch.stack(scores).sum()
        print(expectation)
        

        expectation.backward()
        self.optimizer.step()
        

        self.probs = []
        self.rewards = []
        


        

        


def process(ob):
    #ob = Image.fromarray(ob.astype('uint8'), 'RGB')
    #resize = transforms.Resize(64, 64)
    #ob = resize(ob)
    ob = torch.FloatTensor(ob.reshape(1, 3, 210, 160))
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

 


