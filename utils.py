import torch
from torch import nn

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(-1)
    
# https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/master/a2c_ppo_acktr/utils.py#L32
class AddBias(nn.Module):
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x):
        # if x.dim() == 2:
        bias = self._bias.t().view(1, -1)
        # else: #idk wtf is happening
        #     bias = self._bias.t().view(1, -1, 1, 1)

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
