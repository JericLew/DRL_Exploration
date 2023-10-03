import torch
import ray
from parameter import *
from worker import Worker
from network import RL_Policy


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

    def do_job(self, curr_episode):
        save_img = True and GLOBAL_SAVE_IMG if curr_episode % SAVE_IMG_GAP == 0 else False
        worker = Worker(self.meta_agent_id, self.actor_critic, curr_episode, save_image=save_img)
        worker.work(curr_episode)

        job_results = worker.episode_buffer
        perf_metrics = worker.perf_metrics

        return job_results, perf_metrics

    def job(self, weights, episode_number):
        print("starting episode {} on metaAgent {}".format(episode_number, self.meta_agent_id))
        # set the local weights to the global weight values from the master network
        self.set_actor_critic_weights(weights)

        job_results, metrics = self.do_job(episode_number)

        info = {
            "id": self.meta_agent_id,
            "episode_number": episode_number,
        }

        return job_results, metrics, info

  
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
