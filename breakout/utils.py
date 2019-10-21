import torch
import numpy as np
import cv2

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
 
