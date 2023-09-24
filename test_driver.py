import ray
import numpy as np
import os
import torch

from network import RL_Policy
from test_worker import TestWorker
from test_parameter import *


def run_test():
    if not os.path.exists(trajectory_path):
        os.makedirs(trajectory_path)

    # Handle devices for global training and local simulation
    device = torch.device('cuda') if USE_GPU else torch.device('cpu')

    # initialize actor critic network
    actor_critic = RL_Policy(INPUT_DIM, 2).to(device)

    if device == 'cuda':
        checkpoint = torch.load(f'{model_path}/checkpoint.pth')
    else:
        checkpoint = torch.load(f'{model_path}/checkpoint.pth', map_location = torch.device('cpu'))

    actor_critic.load_state_dict(checkpoint['policy_model'])

    meta_agents = [Runner.remote(i) for i in range(NUM_META_AGENT)]
    weights = actor_critic.state_dict()
    curr_test = 0

    dist_history = []

    job_list = []
    for i, meta_agent in enumerate(meta_agents):
        job_list.append(meta_agent.job.remote(weights, curr_test))
        curr_test += 1

    try:
        while len(dist_history) < curr_test:
            done_id, job_list = ray.wait(job_list)
            done_jobs = ray.get(done_id)

            for job in done_jobs:
                metrics, info = job
                dist_history.append(metrics['travel_dist'])
            if curr_test < NUM_TEST:
                job_list.append(meta_agents[info['id']].job.remote(weights, curr_test))
                curr_test += 1

        print('|#Total test:', NUM_TEST)
        print('|#Average length:', np.array(dist_history).mean())
        print('|#Length std:', np.array(dist_history).std())

    except KeyboardInterrupt:
        print("CTRL_C pressed. Killing remote workers")
        for a in meta_agents:
            ray.kill(a)


@ray.remote(num_cpus=1, num_gpus=NUM_GPU/NUM_META_AGENT)
class Runner(object):
    def __init__(self, meta_agent_id):
        self.meta_agent_id = meta_agent_id
        self.device = torch.device('cuda') if USE_GPU else torch.device('cpu')
        self.actor_critic = RL_Policy(INPUT_DIM, 2).to(self.device)

    def get_weights(self):
        return self.actor_critic.state_dict()
    
    def set_actor_critic_weights(self, weights):
        self.actor_critic.load_state_dict(weights)

    def do_job(self, curr_episode):
        worker = TestWorker(self.meta_agent_id, self.actor_critic, curr_episode, save_image=SAVE_GIFS)
        worker.work(curr_episode)

        perf_metrics = worker.perf_metrics
        return perf_metrics

    def job(self, weights, episode_number):
        print("starting episode {} on metaAgent {}".format(episode_number, self.meta_agent_id))
        # set the local weights to the global weight values from the master network
        self.set_actor_critic_weights(weights)

        metrics = self.do_job(episode_number)

        info = {
            "id": self.meta_agent_id,
            "episode_number": episode_number,
        }

        return metrics, info


if __name__ == '__main__':
    ray.init()
    for i in range(NUM_RUN):
        run_test()
