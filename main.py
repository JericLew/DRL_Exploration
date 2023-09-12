import torch
import sys

from parameter import *
from ppo import PPO

def train():
    print(f"Training", flush=True)

	# Create a model for PPO.s
    model = PPO()

    # Tries to load in an existing actor/critic model to continue training on
    # if actor_model != '' and critic_model != '':
    #     print(f"Loading in {actor_model} and {critic_model}...", flush=True)
    #     model.actor.load_state_dict(torch.load(actor_model))
    #     model.critic.load_state_dict(torch.load(critic_model))
    #     print(f"Successfully loaded.", flush=True)
    # elif actor_model != '' or critic_model != '': # Don't train from scratch if user accidentally forgets actor/critic model
    #     print(f"Error: Either specify both actor/critic models or none at all. We don't want to accidentally override anything!")
    #     sys.exit(0)
    # else:
    print(f"Training from scratch.", flush=True)

    model.learn(1000000)

train()