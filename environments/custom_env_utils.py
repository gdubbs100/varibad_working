"""
Based on https://github.com/ikostrikov/pytorch-a2c-ppo-acktr
"""
import gym
import torch

from environments.env_utils.vec_env import VecEnvWrapper
from environments.env_utils.vec_env.subproc_vec_env import SubprocVecEnv

def make_continual_env(env_id, **kwargs):
    def _thunk():
        env = gym.make(env_id, **kwargs)
        ## TODO: may need to add wrappers here?
        return env
    return _thunk

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
    
