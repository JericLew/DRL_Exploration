import torch
from torch import nn

from parameter import *
    
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
        self.linear2 = nn.Linear(hidden_size, 256)
        self.critic_linear = nn.Linear(256, 1)
        self.train()

    def forward(self, inputs):
        # print(inputs.shape)
        x = self.main(inputs)
        # print(x.shape)
        x = nn.ReLU()(self.linear1(x))
        x = nn.ReLU()(self.linear2(x))
        return self.critic_linear(x).squeeze(-1), x
        # squeeze -1 removes dimension of size 1 along last axis of tensor to remove singleton


class RL_Policy(nn.Module):

    def __init__(self, obs_shape, action_dim):
        super(RL_Policy, self).__init__()
        self.network = Global_Policy(obs_shape,hidden_size=HIDDEN_SIZE)

        self.action_dim = action_dim
        self.dist = DiagGaussian(self.network.output_size, action_dim)

    def forward(self, observations):
        return self.network(observations)
    
    def act(self, observations):
        with torch.no_grad():
            value, actor_features = self(observations.unsqueeze(0)) #add batch dimension
            dist = self.dist(actor_features)
            action = dist.sample().squeeze() # squeeze because it was made for multibatch input
            action_log_probs = dist.log_probs(action).squeeze()
            # print(f"action {action}")
            # print(f"logprobs {action_log_probs}")
        return value, action.detach(), action_log_probs.detach()

    def get_value(self, batch_obs):
        value, _, _ = self(batch_obs)
        return value
    
    def evaluate(self, batch_obs, batch_acts):
        value, actor_features = self(batch_obs)
        dist = self.dist(actor_features)
        action_log_probs = dist.log_probs(batch_acts)
        dist_entropy = dist.entropy().mean()
        # Return the value vector V of each observation in the batch
        # and log probabilities log_probs of each action in the batch
        return value, action_log_probs, dist_entropy

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)
    
# https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/master/a2c_ppo_acktr/utils.py#L32
class AddBias(nn.Module):
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x):
        if x.dim() == 2:
            bias = self._bias.t().view(1, -1)
        else: #idk wtf is happening
            bias = self._bias.t().view(1, -1, 1, 1)

        return x + bias
    

'''
Distribution Stuff
'''

FixedNormal = torch.distributions.Normal
log_prob_normal = FixedNormal.log_prob
FixedNormal.log_probs = lambda self, actions: \
    log_prob_normal(self, actions).sum(-1, keepdim=False)

entropy = FixedNormal.entropy
FixedNormal.entropy = lambda self: entropy(self).sum(-1) # total entropy

FixedNormal.mode = lambda self: self.mean

class DiagGaussian(nn.Module):

    def __init__(self, num_inputs, num_outputs):
        super(DiagGaussian, self).__init__()

        self.fc_mean = nn.Linear(num_inputs, num_outputs)
        self.logstd = AddBias(torch.zeros(num_outputs))

    def forward(self, x):
        action_mean = self.fc_mean(x)
        zeros = torch.zeros(action_mean.size())
        if x.is_cuda:
            zeros = zeros.cuda()

        action_logstd = self.logstd(zeros)
        return FixedNormal(action_mean, action_logstd.exp())
