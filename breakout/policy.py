import torch.nn as nn
from collections import OrderedDict
import torch




class PolicyNet(nn.Module):
    def __init__(self, feature_shape, action_space):
        super().__init__()
        self.action_space = action_space
        self.state_shape = feature_shape

        self.conv_layer = OrderedDict([
                            ('conv1', nn.Conv2d(1, 4, kernel_size=5)),
                            ('relu1', nn.Tanh()),
                            ('conv2', nn.Conv2d(4, 8, kernel_size=5)),
                            ('relu2', nn.Tanh()),
                           
                           ])
        
        self.arch = OrderedDict([
                    ('fc1', nn.Linear(46208, 250)),
                    ('relu1', nn.Tanh()),

                    ('fc2', nn.Linear(250, 100)),
                    ('relu3', nn.Tanh()),
                    ('fc3', nn.Linear(100, 50)),
                    ('relu5', nn.Tanh()),
                    ('fc4', nn.Linear(50, self.action_space))
                    ])

        self.model = nn.Sequential(self.arch)
        self.conv_model = nn.Sequential(self.conv_layer)

    def forward(self, x):
        x = self.conv_model(x)
        x = x.view(x.shape[0], -1)
        
        x = self.model(x)
        return torch.softmax(x, dim=1)

    def __call__(self, x):
        probs = self.forward(x)
        distribution = torch.distributions.Categorical(probs)
        action = distribution.sample().item()
        prob = torch.log(probs[0, action])

        return (action, prob)


if __name__ == '__main__':

    p = PolicyNet(3,3)
    print(p)





