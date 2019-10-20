import torch.nn as nn
from collections import OrderedDict
import torch




class PolicyNet(nn.Module):
    def __init__(self, action_space, feature_shape):
        super().__init__()
        self.action_space = action_space
        self.state_shape = feature_shape

        self.conv_layer = OrderedDict([
                            ('conv1', nn.Conv2d(3, 8, kernel_size=5)),
                            ('relu1', nn.ReLU()),
                            ('conv2', nn.Conv2d(8, 20, kernel_size=5)),
                            ('relu2', nn.ReLU()),
                            ('conv3', nn.Conv2d(20, 30, kernel_size=5)),
                            ('relu3', nn.ReLU()),
                           
                           ])
        
        self.arch = OrderedDict([
                    ('fc1', nn.Linear(155520, 650)),
                    ('relu1', nn.ReLU()),

                    ('fc2', nn.Linear(650, 450)),
                    ('relu2', nn.ReLU()),
                    ('fc3', nn.Linear(450, 250)),
                    ('relu3', nn.ReLU()),
                    ('fc4', nn.Linear(250, 120)),
                    ('relu4', nn.ReLU()),
                    ('fc5', nn.Linear(120, 50)),
                    ('relu5', nn.ReLU()),
                    ('fc6', nn.Linear(50, self.action_space))
                    ])

        self.model = nn.Sequential(self.arch)
        self.conv_model = nn.Sequential(self.conv_layer)

    def forward(self, x):
        x = self.conv_model(x)
        x = x.view(x.shape[0], -1)
        
        x = self.model(x)
        return torch.log_softmax(x, dim=1)



