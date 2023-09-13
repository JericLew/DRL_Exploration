import torch
from torch import nn
from utils import Flatten
    
# Global Policy model code
class Global_Policy(nn.Module):
# we removed orientation
    def __init__(self, input_shape, hidden_size=512, downscaling=1):
        super(Global_Policy, self).__init__()
        self.output_size = 256 # hidden_size which is always 256 as per args
         # idk why but for dist
         
        out_size = int(input_shape[1] / 16. * input_shape[2] / 16.)

        self.main = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(8, 32, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, stride=1, padding=1),
            nn.ReLU(),
            Flatten()
        )

        self.linear1 = nn.Linear(out_size * 32, hidden_size)
        self.linear2 = nn.Linear(hidden_size, 2)
        self.critic_linear = nn.Linear(2, 1)
        self.train()

    def forward(self, inputs):
        # print(inputs.shape)
        x = self.main(inputs)
        # print(x.shape)
        x = nn.ReLU()(self.linear1(x))
        x = nn.ReLU()(self.linear2(x))
        return self.critic_linear(x).squeeze(-1), x
        # squeeze -1 removes dimension of size 1 along last axis of tensor to remove singleton
