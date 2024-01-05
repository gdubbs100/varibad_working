"""
Based on https://github.com/ikostrikov/pytorch-a2c-ppo-acktr
"""
import gym
import torch

from continualworld_utils.wrappers import RandomizationWrapper
from continualworld_utils.utils import get_subtasks
from continualworld_utils.constants import MT50
from environments.env_utils.vec_env import VecEnvWrapper
from environments.env_utils.vec_env.subproc_vec_env import SubprocVecEnv
from environments.parallel_envs import VecNormalize


def make_continual_env(env_id, **kwargs):
    def _thunk():
        env = gym.make(env_id, **kwargs)
        return env
    return _thunk

def prepare_base_envs(
        task_names,
        benchmark = MT50,
        task_set = 'train', 
        randomization="random_init_fixed20",
        # normalise_rewards = True
        ):
    """
    task_names: list of task names from metworld benchmark
    benchmark: a set of metaworld benchmark tasks (default is MT50)
    randomization: string to pass to randomization_wrapper
    """
    envs = []
    for task_name in task_names:
        
        if task_set=='train':
            env = benchmark.train_classes[task_name]()
        elif task_set=='test':
            env=benchmark.test_classes[task_name]()
        else:
            raise ValueError('task_set must be one of test or train')
        # if normalise_rewards:
        #     envs = VecNormalize(envs, normalise_rew=normalise_rewards, ret_rms=None, gamma=gamma)
        env = RandomizationWrapper(env, get_subtasks(task_name, benchmark, task_set), randomization)
        env.name = task_name
        envs.append(env)
    return envs

def prepare_parallel_envs(envs, steps_per_env, num_processes, device):
    subproc_envs = SubprocVecEnv(
        [make_continual_env('continualMW-v0', **{'envs' : envs, 'steps_per_env': steps_per_env}) for _ in range(num_processes)]
    )
    pytorch_envs = PyTorchVecEnvCont(subproc_envs, device)
    return pytorch_envs

class PyTorchVecEnvCont(VecEnvWrapper):

    def __init__(self, vec_envs, device):
        super(PyTorchVecEnvCont, self).__init__(vec_envs)
        self.device = device
  
    def step_async(self, actions):
        actions = actions.squeeze().cpu().numpy()
        self.venv.step_async(actions)

    def step_wait(self):
        state, reward, done, info = self.venv.step_wait()
        if isinstance(state, list):  # raw + normalised .permute(1, 0, 2)
            state = [torch.from_numpy(s).permute(1, 0, 2).float().to(self.device) for s in state]
        else:
            state = torch.from_numpy(state).permute(1, 0, 2).float().to(self.device)
        # reshape rewards to have dim T X B X D .reshape(1, -1, 1)
        if isinstance(reward, list):  # raw + normalised
            reward = [torch.from_numpy(r).unsqueeze(dim=1).reshape(1, -1, 1).float().to(self.device) for r in reward]
        else:
            reward = torch.from_numpy(reward).unsqueeze(dim=1).reshape(1, -1, 1).float().to(self.device)
        return state, reward, done, info
    
    def reset(self):
        # if task is not None:
        #     assert isinstance(task, list)
        state = self.venv.reset()
        ## permute state to have dimensions T X B X D .permute(1,0,2)
        if isinstance(state, list):
            state = [torch.from_numpy(s).permute(1, 0, 2).float().to(self.device) for s in state]
        else:
            state = torch.from_numpy(state).permute(1, 0, 2).float().to(self.device)
        return state

    def __getattr__(self, attr):
        """ If env does not have the attribute then call the attribute in the wrapped_env """

        if attr in ['_max_episode_steps', 'task_dim', 'belief_dim', 'num_states']:
            return self.unwrapped.get_env_attr(attr)

        try:
            orig_attr = self.__getattribute__(attr)
        except AttributeError:
            orig_attr = self.unwrapped.__getattribute__(attr)

        if callable(orig_attr):
            def hooked(*args, **kwargs):
                result = orig_attr(*args, **kwargs)
                return result

            return hooked
        else:
            return orig_attr
    
