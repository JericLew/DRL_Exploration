import torch
import ray
from network import RL_Policy
from worker import Worker
from parameter import *


class Runner(object):
    def __init__(self, meta_agent_id):
        self.meta_agent_id = meta_agent_id
        self.local_device = torch.device('cuda') if USE_GPU else torch.device('cpu')
        # Initialise local actor critic for simulation
        self.actor_critic = RL_Policy(INPUT_DIM, 2).to(self.local_device)

    def get_weights(self):
        return self.actor_critic.state_dict()

    def set_actor_critic_weights(self, weights):
        self.actor_critic.load_state_dict(weights)

    def do_job(self, epi_so_far):
        save_img = True if epi_so_far % SAVE_IMG_GAP == 0 else False
        save_img = save_img and GLOBAL_SAVE_IMG
        worker = Worker(self.meta_agent_id, epi_so_far, self.actor_critic, save_image=save_img)
        worker.work(epi_so_far)

        # Initialise batch data
        batch_obs = []
        batch_acts = []
        batch_rewards = []
        batch_returns = []
        batch_episode_len = []

        # Append episode data
        with torch.no_grad():  
            batch_obs.append(torch.stack(worker.episode_obs))
            batch_acts.append(torch.stack(worker.episode_acts))
            batch_rewards.append(worker.episode_rewards)
            batch_episode_len.append(worker.episode_len)

        # Reshape data as tensors in the shape specified in function description, before returning
        batch_obs = torch.cat(batch_obs).to(self.device)
        batch_acts = torch.cat(batch_acts).to(self.device)
        batch_returns = self.compute_returns(batch_rewards)

        # Log the episodic returns and episodic lengths in this batch.
        self.logger['batch_rews'] = batch_rewards
        self.logger['batch_lens'] = batch_episode_len

        return batch_obs, batch_acts, batch_returns, batch_episode_len, epi_so_far

    def job(self, weights, episode_number):
        print("starting episode {} on metaAgent {}".format(episode_number, self.meta_agent_id))
        # set the local weights to the global weight values from the master network
        self.set_actor_critic_weights(weights)

        batch_obs, batch_acts, batch_returns, batch_episode_len, epi_so_far = self.do_job(episode_number)

        info = {
            "id": self.meta_agent_id,
            "episode_number": episode_number,
        }

        return batch_obs, batch_acts, batch_returns, batch_episode_len, epi_so_far, info

  
@ray.remote(num_cpus=1, num_gpus=NUM_GPU/NUM_META_AGENT)
class RLRunner(Runner):
    def __init__(self, meta_agent_id):        
        super().__init__(meta_agent_id)


if __name__=='__main__':
    ray.init()
    runner = RLRunner.remote(0)
    job_id = runner.do_job.remote(1)
    out = ray.get(job_id)
    print(out[1])
