import os
import time

import gym
import numpy as np
import torch

from copy import deepcopy

from models.combined_actor_critic import ActorCritic, BiHemActorCritic
from models.policy import Policy
from models.encoder import RNNEncoder

from algorithms.custom_ppo import CustomPPO, BiHemPPO
from algorithms.custom_storage import CustomOnlineStorage, BiHemOnlineStorage

from utils import helpers as utl
from utils.custom_helpers import get_args_from_config, freeze_parameters
from utils.custom_logger import CustomLogger
from environments.custom_env_utils import prepare_parallel_envs, prepare_base_envs
from environments.custom_metaworld_benchmark import ML3

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ContinualLearner:
    """
    Continual learning class - handles training process for continual learning
    """
    def __init__(
            self, 
            seed, 
            task_names,
            num_processes, 
            rollout_len, 
            steps_per_env, 
            normalise_rewards,
            log_dir,  
            gamma = 0.99,
            tau = 0.95,
            eval_every = 10,
            quantiles = [i/10 for i in range(1, 10)],
            randomization = 'random_init_fixed20',
            args = None):

        self.args = args

        ## TODO: set a seed, look at below function
        utl.seed(seed, False)
        self.seed = seed

        self.gamma = gamma
        self.tau = tau
        self.normalise_rewards = normalise_rewards

        ## initialise the envs
        self.raw_train_envs = prepare_base_envs(
            task_names, 
            benchmark=ML3(),
            task_set = self.args.task_set,#'test', # we train on the test set of ML3 for bicameral
            randomization=randomization)

        ## get unique task names in order:
        _, idx = np.unique(task_names, return_index=True)
        self.task_names = [task_names[i] for i in np.sort(idx)]

        self.env_id_to_name = {(i+1):task for i, task in enumerate(self.task_names)}
        self.envs = prepare_parallel_envs(
            envs = self.raw_train_envs,
            steps_per_env=steps_per_env,
            num_processes=num_processes,
            seed = self.seed,
            gamma=self.gamma,
            normalise_rew=self.normalise_rewards,
            device=device
        )

        # set params for runs
        self.num_processes = num_processes
        self.rollout_len = rollout_len

        # create network and agent
        self.agent, self.left_init_args, self.right_init_args = self.init_agent(self.args)
        if self.args.algorithm == 'random':
            self.storage = None
        elif self.args.algorithm != 'bicameral':
            self.storage = CustomOnlineStorage(
                        self.rollout_len, 
                        self.num_processes, 
                        self.envs.observation_space.shape[0]+1, 
                        0, # what's this? 
                        0, # what's this?
                        self.envs.action_space, 
                        self.agent.actor_critic.encoder.hidden_size, 
                        self.agent.actor_critic.encoder.latent_dim, 
                        self.normalise_rewards # normalise rewards for policy - set to true, but implement
                    )
        elif self.args.algorithm == 'bicameral':
                self.storage = BiHemOnlineStorage(
                    self.rollout_len, 
                    self.num_processes, 
                    self.envs.observation_space.shape[0]+1, 
                    0, # what's this? 
                    0, # what's this?
                    self.envs.action_space, 
                    ## BiHemOnlineStorage - have separate left / right encoder/ hidden dims
                    gate_hidden_size=self.agent.actor_critic.gating_network.hidden_size,
                    left_hidden_size=self.agent.actor_critic.left_actor_critic.encoder.hidden_size, 
                    right_hidden_size=self.agent.actor_critic.right_actor_critic.encoder.hidden_size, 
                    gate_latent_dim=self.agent.actor_critic.gating_network.latent_dim,
                    left_latent_dim=self.agent.actor_critic.left_actor_critic.encoder.latent_dim, 
                    right_latent_dim=self.agent.actor_critic.right_actor_critic.encoder.latent_dim, 
                    normalise_rewards=self.normalise_rewards # normalise rewards for policy - set to true, but implement
                )
        
        self.quantiles = quantiles
        self.log_dir = log_dir
        self.logger = CustomLogger(
            self.log_dir, 
            self.quantiles, 
            args = self.args, 
            left_args = self.left_init_args,
            right_args = self.right_init_args)
        self.eval_every = eval_every

    def init_agent(self, args):
        ## update if relevant
        left_init_args = None
        right_init_args = None

        ## create agent / actor critic
        if self.args.algorithm != 'right_only':
            ## create randomly initialised policy with settings from config file
            left_init_args = get_args_from_config(self.args.run_folder)
            left_policy_net = Policy(
                args=left_init_args,
                pass_state_to_policy=left_init_args.pass_state_to_policy,
                pass_latent_to_policy=left_init_args.pass_latent_to_policy,
                pass_belief_to_policy=left_init_args.pass_belief_to_policy,
                pass_task_to_policy=left_init_args.pass_task_to_policy,
                dim_state=self.envs.observation_space.shape[0]+1, # to add done flag
                dim_latent=left_init_args.latent_dim * 2,
                dim_belief=0,
                dim_task=0,
                hidden_layers=left_init_args.policy_layers,
                activation_function=left_init_args.policy_activation_function,
                policy_initialisation=left_init_args.policy_initialisation,
                action_space=self.envs.action_space,
                init_std=left_init_args.policy_init_std
            ).to(device)

            left_encoder_net = RNNEncoder(
                args=left_init_args,
                layers_before_gru=left_init_args.encoder_layers_before_gru,
                hidden_size=left_init_args.encoder_gru_hidden_size,
                layers_after_gru=left_init_args.encoder_layers_after_gru,
                latent_dim=left_init_args.latent_dim,
                action_dim=self.envs.action_space.shape[0],
                action_embed_dim=left_init_args.action_embedding_size,
                state_dim=self.envs.observation_space.shape[0]+1, # for done flag
                state_embed_dim=left_init_args.state_embedding_size,
                reward_size=1,
                reward_embed_size=left_init_args.reward_embedding_size,
            ).to(device)

        if self.args.algorithm != 'left_only':
            policy_loc = args.run_folder + '/models/policy.pt'
            encoder_loc = args.run_folder + '/models/encoder.pt'
            right_policy_net = torch.load(policy_loc, map_location=device).to(device)
            right_encoder_net = torch.load(encoder_loc, map_location=device).to(device)
            ## freeze these!
            freeze_parameters(right_policy_net)
            freeze_parameters(right_encoder_net)

            ## create init_args from loaded networks
            assert right_policy_net.args==right_encoder_net.args, "policy and encoder args should match!"
            right_init_args = right_policy_net.args
            del right_init_args.action_space # not needed for logs, causes error in json
        
        if self.args.algorithm == 'bicameral':
            ac = BiHemActorCritic(
                left_policy_net, left_encoder_net,
                right_policy_net, right_encoder_net,
                self.envs.observation_space.shape[0] + 1, 
                self.envs.action_space.shape[0],
                init_std = args.init_std,
                ## gating encoder args
                use_action_in_gate = self.args.use_action_in_gate,
                use_state_in_gate = self.args.use_state_in_gate,
                ## gating schedule functions
                use_gating_schedule = self.args.use_gating_schedule,
                gating_schedule_type = self.args.gating_schedule_type,
                gating_schedule_update = self.args.gating_schedule_update,
                min_right_value=self.args.min_right_value,
                init_right_value = self.args.init_right_value
            ).to(device)

            agent = BiHemPPO(
                actor_critic=ac,
                value_loss_coef = self.args.value_loss_coef,
                entropy_coef = self.args.entropy_coef,
                policy_optimiser=self.args.optimiser,
                lr = self.args.learning_rate,
                eps= self.args.eps, # for adam optimiser
                clip_param = self.args.ppo_clip_param, 
                ppo_epoch = self.args.ppo_epoch, 
                num_mini_batch=self.args.num_mini_batch,
                use_huber_loss = self.args.use_huberloss,
                use_clipped_value_loss=self.args.use_clipped_value_loss,
                use_gating_penalty = self.args.use_gating_penalty,
                gating_alpha=self.args.gating_alpha,
                gating_beta=self.args.gating_beta,
                context_window=self.args.context_window
            )
        elif self.args.algorithm == 'left_only':
            ac = ActorCritic(left_policy_net, left_encoder_net)
            agent = CustomPPO(
                actor_critic=ac,
                value_loss_coef = self.args.value_loss_coef,
                entropy_coef = self.args.entropy_coef,
                policy_optimiser=self.args.optimiser,
                lr = self.args.learning_rate,
                eps= self.args.eps, # for adam optimiser
                clip_param = self.args.ppo_clip_param, 
                ppo_epoch = self.args.ppo_epoch, 
                num_mini_batch=self.args.num_mini_batch,
                use_huber_loss = self.args.use_huberloss,
                use_clipped_value_loss=self.args.use_clipped_value_loss,
                context_window=self.args.context_window
            )
        elif self.args.algorithm == 'right_only':
            ac = ActorCritic(right_policy_net, right_encoder_net)
            agent = CustomPPO(
                actor_critic=ac,
                value_loss_coef = self.args.value_loss_coef,
                entropy_coef = self.args.entropy_coef,
                policy_optimiser=self.args.optimiser,
                lr = self.args.learning_rate,
                eps= self.args.eps, # for adam optimiser
                clip_param = self.args.ppo_clip_param, 
                ppo_epoch = self.args.ppo_epoch, 
                num_mini_batch=self.args.num_mini_batch,
                use_huber_loss = self.args.use_huberloss,
                use_clipped_value_loss=self.args.use_clipped_value_loss,
                context_window=self.args.context_window
            )
        elif self.args.algorithm == 'random':
            agent, left_init_args, right_init_args = None, None, None
        else:
            raise NotImplementedError("Set algorithm to one of left_only, right_only, bicameral or random")

        

        return agent, left_init_args, right_init_args
    
    def train(self):
        """ Main Training loop """
        start_time = time.time() 
        eps = 0

        # steps limit is parameter for whole continual env
        while self.envs.get_env_attr('cur_step') < self.envs.get_env_attr('steps_limit'):

            step = 0
            obs = self.envs.reset() # we reset all at once as metaworld is time limited
            current_task = self.envs.get_env_attr("cur_seq_idx")
            episode_reward = []
            successes = []
            gating_values = []
            done = [False for _ in range(self.num_processes)]

            if self.args.algorithm != 'random':
                with torch.no_grad():

                    latent, hidden_state = self.agent.get_prior(self.num_processes)
                    # assert len(self.storage.latent) == 0  # make sure we emptied buffers

                    if self.args.algorithm == 'bicameral':
                        assert (len(self.storage.left_latent) == 0) & (len(self.storage.right_latent) == 0)

                        self.storage.gate_hidden_states[:1].copy_(hidden_state[0])
                        self.storage.left_hidden_states[:1].copy_(hidden_state[1])
                        self.storage.right_hidden_states[:1].copy_(hidden_state[2])
                        self.storage.gate_latent.append(latent[0])
                        self.storage.left_latent.append(latent[1])
                        self.storage.right_latent.append(latent[2])
                    else:
                        assert len(self.storage.latent) == 0  # make sure we emptied buffers
                        self.storage.hidden_states[:1].copy_(hidden_state)
                        self.storage.latent.append(latent)

            while not all(done):
                with torch.no_grad():
                    if self.args.algorithm == 'bicameral':
                        ## TODO: don't like unsqueeze obs but ok for now
                        (value, left_value, right_value), action, gate_values = \
                            self.agent.act(obs.unsqueeze(0), latent, None, None)
                        ## collect gating values
                        gating_values.append(gate_values[0].detach())
                    elif self.args.algorithm == 'random':
                        action = torch.tensor(
                            np.array(
                                [self.envs.action_space.sample() for _ in range(self.num_processes)]
                            )
                        )
                        gating_values.append(torch.tensor(0.))
                    elif self.args.algorithm == 'right_only':
                        value, action = self.agent.act(obs, latent, None, None, deterministic=True)
                        ## dummy gating value
                        gating_values.append(torch.tensor(0.))
                    else:
                        value, action = self.agent.act(obs, latent, None, None)
                        ## dummy gating value
                        gating_values.append(torch.tensor(0.))

                next_obs, (rew_raw, rew_normalised), done, info = self.envs.step(action)
                assert all(done) == any(done), "Metaworld envs should all end simultaneously"
                
                ## calculate value errors for left/right
                if self.args.algorithm == 'bicameral':
                    value_errors = (
                        rew_raw - left_value,
                        rew_raw - right_value
                    )

                # create mask for episode ends
                masks_done = torch.FloatTensor([[0.0] if _done else [1.0] for _done in done]).to(device)

                ## combine all rewards
                episode_reward.append(rew_raw)
                # if we succeed at all then the task is successful
                successes.append(torch.tensor([i['success'] for i in info]))
                if self.args.algorithm != 'random':
                    with torch.no_grad():
                        if self.args.algorithm == 'bicameral':
                            latent, hidden_state = self.agent.get_latent(
                                action, next_obs, rew_raw, 
                                value_errors, gate_values, hidden_state, 
                                return_prior = False
                            )
                        else:
                            latent, hidden_state = self.agent.get_latent(
                                action, next_obs, rew_raw, hidden_state, return_prior = False
                            )
                    
                    self.storage.next_state[step] = next_obs.clone()

                    if self.args.algorithm != 'bicameral':
                        self.storage.insert(
                            state=next_obs.squeeze(),
                            belief=None, # could I get rid of belief?
                            task=None, # could I get rid of task?
                            actions=action.double(),
                            rewards_raw=rew_raw.squeeze(0),
                            rewards_normalised=rew_normalised.squeeze(0),
                            value_preds= value.squeeze(0),
                            masks=masks_done.squeeze(0), 
                            done=torch.from_numpy(done)[:,None].float(),
                            hidden_states = hidden_state.squeeze(),
                            latent = latent,
                        )
                        
                    else:
                        # hidden state is tuple
                        self.storage.insert(
                            state=next_obs.squeeze(),
                            belief=None, # could I get rid of belief?
                            task=None, # could I get rid of task?
                            actions=action.double(),
                            rewards_raw=rew_raw.squeeze(0),
                            rewards_normalised=rew_normalised.squeeze(0),
                            value_preds=(value.squeeze(0), left_value.squeeze(0), right_value.squeeze(0)),
                            masks=masks_done.squeeze(0), 
                            done=torch.from_numpy(done)[:,None].float(),
                            hidden_states = hidden_state,
                            latent = latent,
                        )
   
                obs = next_obs

                step += 1
            if self.args.algorithm != 'random':
                with torch.no_grad():
                    if self.args.algorithm=='bicameral':
                        ## BUG: next obs vs obs - should be the same at this point, but not good
                        latent, hidden_state = self.agent.get_latent(
                                action, obs, rew_raw, 
                                value_errors, gate_values, hidden_state, 
                                return_prior = False
                            )
                        
                        ## only need final value for returns
                        value, _, _ = self.agent.get_value(
                            obs.unsqueeze(0),
                            latent,
                            None,
                            None
                        )

                    else:
                        latent, hidden_state = self.agent.get_latent(
                            action, obs, rew_raw, hidden_state, return_prior = False
                        )

                        value = self.agent.get_value(
                            obs.unsqueeze(0),
                            latent,
                            None,
                            None
                        )

                # compute returns - use_proper_time_limits is false
                self.storage.compute_returns(
                    next_value = value.detach(), # detach from computation graph
                    use_gae = True,
                    gamma = self.gamma,
                    tau = self.tau,
                    use_proper_time_limits=False
                )

            ## Update
            if self.args.algorithm == 'bicameral':
                value_loss_epoch, action_loss_epoch, dist_entropy_epoch, gating_penalty_epoch, loss_epoch = \
                    self.agent.update(self.storage)
            elif self.args.algorithm == 'left_only':
                value_loss_epoch, action_loss_epoch, dist_entropy_epoch, loss_epoch = \
                    self.agent.update(self.storage)
                gating_penalty_epoch = np.nan
            else:
                value_loss_epoch, action_loss_epoch, dist_entropy_epoch, gating_penalty_epoch, loss_epoch = \
                    np.nan, np.nan, np.nan, np.nan, np.nan

            ## calculate environment steps 
            frames = (eps+1) * self.num_processes * self.rollout_len
            ## log training loss
            self.logger.add_tensorboard('losses/value_loss', value_loss_epoch, frames)
            self.logger.add_tensorboard('losses/action_loss', action_loss_epoch, frames)
            self.logger.add_tensorboard('losses/entropy_loss', dist_entropy_epoch, frames)
            self.logger.add_tensorboard('losses/gating_penalty', gating_penalty_epoch, frames)
            self.logger.add_tensorboard('losses/total_loss', loss_epoch, frames)
            
            # log training results
            task_rewards = torch.stack(episode_reward).cpu()
            task_successes = torch.stack(successes).max(0)[0].mean()
            task_gating_values = torch.stack(gating_values).cpu()
            self.logger.add_tensorboard('train_results/episode_rewards',task_rewards.mean(), frames)
            self.logger.add_tensorboard('train_results/episode_success',task_successes, frames)
            self.logger.add_tensorboard('train_results/left_gating_values', task_gating_values.mean(), frames)

            self.logger.add_tensorboard('current_task', current_task, frames)

            ## save to csv
            self.log_results(
                self.env_id_to_name[current_task + 1], 
                task_rewards,
                task_successes,
                task_gating_values,
                self.num_processes,
                self.env_id_to_name[current_task + 1],
                frames,
                'train')
            
            if self.storage is not None:
                # clears out old data
                self.storage.after_update()

            if (eps+1) % self.eval_every == 0:

                if self.args.algorithm not in ['right_only', 'random']:
                    print(f"Running eval on full model at {eps + 1}")
                    ## run eval on full network
                    self.evaluate(current_task, frames, 'test')
                    

                if self.args.algorithm == 'bicameral':
                    print(f"Running eval on left only at {eps + 1}")
                    ## run eval on left network
                    self.evaluate(current_task, frames, 'left')
                    ## run eval on right network
                    # self.evaluate(current_task, frames, 'right')

                if self.args.algorithm != 'random':
                    ## save the network
                    self.logger.save_network(self.agent.actor_critic)

            # gating network stepper
            if (self.args.use_gating_schedule) and ((eps+1) % self.args.step_gate_every == 0):
                 self.agent.actor_critic.gating_network.step()

            eps+=1
        end_time = time.time()
        print(f"completed in {end_time - start_time}")
        self.envs.close()

    def evaluate(self, current_task, frames, eval_run):
        
        ## create agent
        if eval_run == 'left':
            ac = self.agent.actor_critic.left_actor_critic
            eval_agent = CustomPPO(
                    actor_critic=ac,
                    value_loss_coef = self.args.value_loss_coef,
                    entropy_coef = self.args.entropy_coef,
                    policy_optimiser=self.args.optimiser,
                    lr = self.args.learning_rate,
                    eps= self.args.eps, # for adam optimiser
                    clip_param = self.args.ppo_clip_param, 
                    ppo_epoch = self.args.ppo_epoch, 
                    num_mini_batch=self.args.num_mini_batch,
                    use_huber_loss = self.args.use_huberloss,
                    use_clipped_value_loss=self.args.use_clipped_value_loss,
                    context_window=self.args.context_window
                )
        elif eval_run == 'test':
            eval_agent = deepcopy(self.agent)
        else:
            raise NotImplementedError
            eval_run = 'right'
            ac = deepcopy(self.agent.actor_critic.right_actor_critic)


        ## create eval environments
        raw_test_envs = prepare_base_envs(
            self.task_names, 
            benchmark=ML3(),
            task_set = self.args.task_set,
        )
        test_envs = prepare_parallel_envs(
            envs=raw_test_envs, 
            steps_per_env=self.rollout_len,
            num_processes=self.num_processes,
            gamma=self.gamma,
            seed=self.seed,
            normalise_rew=self.normalise_rewards,
            device=device,
            rank_offset=self.num_processes+1 # avoids overwriting training temp files - can be disastrous!
        )
        ## run eval loop
        start_time = time.time() 
        eps = 0

        # steps limit is parameter for whole continual env
        while test_envs.get_env_attr('cur_step') < test_envs.get_env_attr('steps_limit'):

            obs = test_envs.reset() # we reset all at once as metaworld is time limited
            current_task = test_envs.get_env_attr("cur_seq_idx")
            episode_reward = []
            successes = []
            gating_values = []

            done = [False for _ in range(self.num_processes)]

            with torch.no_grad():
                latent, hidden_state = eval_agent.get_prior(self.num_processes)

            while not all(done):
                with torch.no_grad():
                    if (self.args.algorithm == 'bicameral') and (eval_run != 'left'):
                        ## TODO: don't like unsqueeze obs but ok for now
                        (_, left_value, right_value), action, gate_values = \
                            self.agent.act(obs.unsqueeze(0), latent, None, None, deterministic=True)
                        ## collect gating values
                        gating_values.append(gate_values[0].detach())
                    else:
                        _, action = eval_agent.act(obs, latent, None, None, deterministic=True)
                        ## dummy gating value
                        gating_values.append(torch.tensor(0.))

                next_obs, (rew_raw, _), done, info = test_envs.step(action)
                assert all(done) == any(done), "Metaworld envs should all end simultaneously"

                ## combine all rewards
                episode_reward.append(rew_raw)
                # if we succeed at all then the task is successful
                successes.append(torch.tensor([i['success'] for i in info]))

                with torch.no_grad():
                    if (self.args.algorithm == 'bicameral') and (eval_run != 'left'):
                        value_errors = (
                            rew_raw - left_value,
                            rew_raw - right_value
                        )
                        latent, hidden_state = self.agent.get_latent(
                            action, next_obs, rew_raw, 
                            value_errors, gate_values, hidden_state, 
                            return_prior = False
                        )
                    else:
                        latent, hidden_state = eval_agent.get_latent(
                            action, next_obs, rew_raw, hidden_state, return_prior = False
                        )
   
                obs = next_obs

            # log eval results
            task_rewards = torch.stack(episode_reward).cpu()
            task_successes = torch.stack(successes).max(0)[0].mean()
            task_gating_values = torch.stack(gating_values).cpu()
            self.logger.add_tensorboard(f'{eval_run}/episode_rewards',task_rewards.mean(), frames)
            self.logger.add_tensorboard(f'{eval_run}/episode_success',task_successes, frames)
            self.logger.add_tensorboard(f'{eval_run}/left_gating_values', task_gating_values.mean(), frames)

            ## save to csv
            self.log_results(
                self.env_id_to_name[current_task + 1], 
                task_rewards,
                task_successes,
                task_gating_values, 
                self.num_processes,
                self.env_id_to_name[current_task + 1],
                frames,
                eval_run)

            eps+=1
        end_time = time.time()
        print(f"completed in {end_time - start_time}")
        test_envs.close()
        del eval_agent
        del test_envs
        del raw_test_envs


        ## log - with left or right prefix
        ## also create left / right results csvs

    # def evaluate(self, current_task, frames, test_processes = 10):
    #     start_time = time.time() # use this in logger?
    #     current_task_name = self.env_id_to_name[current_task + 1]
    #     print(f"starting evaluation at {start_time} with training task {current_task_name}")
    #     ## num runs given by test_processes
    #     test_envs = prepare_parallel_envs(
    #         envs=self.raw_test_envs, 
    #         steps_per_env=self.rollout_len,
    #         num_processes=test_processes,
    #         gamma=self.gamma,
    #         seed = self.seed,
    #         normalise_rew=self.normalise_rewards,
    #         device=device,
    #         rank_offset=self.num_processes+1 # avoids overwriting training temp files - can be disastrous!
    #     )

    #     # record outputs
    #     task_rewards = {
    #         env.name: [] for env in self.raw_test_envs
    #     }

    #     task_successes = {
    #         env.name: [] for env in self.raw_test_envs
    #     }

    #     task_gating_values = {
    #         env.name: [] for env in self.raw_test_envs
    #     }

    #     while test_envs.get_env_attr('cur_step') < test_envs.get_env_attr('steps_limit'):
    #         current_test_env = test_envs.get_env_attr('cur_seq_idx') + 1
    #         obs = test_envs.reset() # we reset all at once as metaworld is time limited
    #         episode_reward = []
    #         successes = []
    #         gating_values = []
    #         done = [False for _ in range(test_processes)]

    #         ## TODO: determine how frequently to get prior - do at start of each episode for now
    #         with torch.no_grad():
    #             latent, hidden_state = self.agent.get_prior(test_processes)

    #         while not all(done):
    #             # with torch.no_grad():
    #             #     # be deterministic in eval
    #             #     _, action = self.agent.act(obs, latent, None, None, deterministic = True)
    #             with torch.no_grad():
    #                 if self.args.algorithm == 'bicameral':
    #                     value, action, (left_gating_value, _) = self.agent.act(obs.unsqueeze(0), latent, None, None, deterministic=True)
    #                     ## collect gating values
    #                     gating_values.append(left_gating_value.detach())
    #                 else:
    #                     value, action = self.agent.act(obs, latent, None, None, deterministic=True)
    #                     ## dummy gating value
    #                     gating_values.append(torch.tensor(0.))
                
    #             # no need for normalised_reward during eval
    #             next_obs, (rew_raw, _), done, info = test_envs.step(action)
    #             assert all(done) == any(done), "Metaworld envs should all end simultaneously"


    #             ## combine all rewards
    #             episode_reward.append(rew_raw)
    #             # if we succeed at all then the task is successful
    #             successes.append(torch.tensor([i['success'] for i in info]))

    #             with torch.no_grad():
    #                 latent, hidden_state = self.agent.get_latent(
    #                 action, next_obs, rew_raw, hidden_state, return_prior = False
    #                 )

    #             obs = next_obs

    #         ## log the results here
    #         task_rewards[self.env_id_to_name[current_test_env]] = torch.stack(episode_reward).cpu()
    #         task_successes[self.env_id_to_name[current_test_env]] = torch.stack(successes).max(0)[0].mean()
    #         task_gating_values[self.env_id_to_name[current_test_env]] = torch.stack(gating_values).cpu()
  
    #     #log rewards, successes to tensorboard
    #     for task_name in self.env_id_to_name.values():
    #         self.log_results(
    #             task_name, 
    #             task_rewards[task_name],
    #             task_successes[task_name],
    #             task_gating_values[task_name],
    #             test_processes,
    #             current_task_name,
    #             frames,
    #             train=False) # log to test csv

    #     end_time = time.time()
    #     print(f"eval completed in {end_time - start_time}")
    #     test_envs.close()

    def log_results(self, task_name, rewards, successes, gating_values, processes, current_task, frame, csv_to_do):

        ## log csv also
        reward_quantiles = torch.quantile(
            rewards,
            torch.tensor(self.quantiles)
        ).numpy().tolist()

        gating_value_quantiles = torch.quantile(
            gating_values,
            torch.tensor(self.quantiles)
        ).numpy().tolist()

        to_write = (
            current_task,
            task_name,
            successes.numpy(),
            processes, # record number tasks per env
            rewards.mean().numpy(),
            *reward_quantiles,
            *gating_value_quantiles,
            frame
        )
        self.logger.add_csv(to_write, csv_to_do)

