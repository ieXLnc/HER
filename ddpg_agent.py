import os
from torch.optim import Adam
# from Copied_HER.models import actor, critic
from networks import Actor, Critic
from HER.mpi_utils import *
from HER.normalizer import normalizer
from buffers import ReplayBuffer, HERSample
from datetime import datetime

# set GPU for faster training
cuda = torch.cuda.is_available()  # check for CUDA
device = torch.device("cuda" if cuda else "cpu")
print("Job will run on {}".format(device))


class DDPGAgent:
    def __init__(self,
                 env,
                 env_params,
                 params_dict
                 ):
        """
        env_params: env parameters from function get_params (obs shape, goal shape, action shape, action max and name)
        fc_shape: layers nodes
        actor_lr: actor learning rate
        critic_lr: critic learning rate
        replay_strategy: choose replay strategy (for now only 'future')
        replay_k: determine the ratio sample of her idx to take
        clip_range: clip obs in normalizer
        clip_obs: clip obs
        memory_size: memory for the buffer
        batch_size: batch size to take in learning
        n_epochs: number of epoch
        n_cycles: number of time we take the rollouts before learning
        rollout: number of rollouts in one cycle
        n_update: number of time to update the net before soft updating the targets
        test_rollouts: number of test rollouts to evaluate the agent
        noise_eps: amplitude of gaussian noise
        random_eps: epsilon to determine how often the action random is added
        gamma: gamma factor
        tau: tau for soft updating
        action_l2: l2 regression on the actor loss
        """

        self.env = env
        self.env_params = env_params

        # hyperparams
        self.actor_lr = params_dict['actor_lr']
        self.critic_lr = params_dict['critic_lr']
        self.replay_strategy = params_dict['replay_strategy']
        self.replay_k = params_dict['replay_k']

        self.clip_range = params_dict['clip_range']
        self.clip_obs = params_dict['clip_obs']
        self.memory_size = params_dict['memory_size']
        self.batch_size = params_dict['batch_size']
        self.n_epochs = params_dict['n_epochs']
        self.n_cycles = params_dict['n_cycles']
        self.rollout = params_dict['rollout']
        self.n_update = params_dict['n_update']
        self.test_rollouts = params_dict['test_rollouts']
        self.save_models = params_dict['save_models']
        # noise
        self.noise_eps = params_dict['noise_eps']
        self.random_eps = params_dict['random_eps']

        self.gamma = params_dict['gamma']
        self.tau = params_dict['tau']
        self.action_l2 = params_dict['action_l2']

        self.fc = params_dict['fc_shape']

        # create the networks
        self.actor = Actor(env_params, self.fc)
        self.critic = Critic(env_params, self.fc)
        # sync them across cpu
        sync_networks(self.actor)
        sync_networks(self.critic)
        # build targets
        self.actor_target = Actor(env_params, self.fc)
        self.critic_target = Critic(env_params, self.fc)
        # hard update of the target networks
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        # create optim for the networks
        self.actor_optimizer = Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=self.critic_lr)
        # HER and ReplayBuffer
        self.HER = HERSample(self.replay_strategy, self.replay_k, self.env.compute_reward)
        self.buffer = ReplayBuffer(self.env_params, self.memory_size, self.HER.sample_her_transitions)
        # Create normalizer
        self.o_norm = normalizer(size=env_params['obs'], default_clip_range=self.clip_range)
        self.g_norm = normalizer(size=env_params['goal'], default_clip_range=self.clip_range)
        # create the saving dict for the model
        if MPI.COMM_WORLD.Get_rank() == 0:
            if not os.path.exists('./Models'):
                os.mkdir('./Models')
            self.model_path = os.path.join('./Models', env_params['name'])
            if not os.path.exists(self.model_path):
                os.mkdir(self.model_path)

        # Create log
        self.log = {
            'success': [],
            'actor_loss': [],
            'critic_loss': [],
            'epoch': [],
            'mean_actor': [],
            'mean_critic': []
        }

    def learn(self):
        # train the algo
        for epoch in range(self.n_epochs):
            for _ in range(self.n_cycles):
                batch_obs, batch_ag, batch_g, batch_act = [], [], [], []
                for _ in range(self.rollout):
                    # reset rollout
                    obs, achieved_goals, goals, acts = [], [], [], []
                    # reset env
                    observation = self.env.reset()
                    o = observation['observation']
                    ag = observation['achieved_goal']
                    g = observation['desired_goal']
                    # start collecting sample
                    for transition in range(self.env_params['max_timesteps']):
                        with torch.no_grad():
                            action = self.get_action(o, g)             # remove step because gaussian
                        # feed action
                        new_observation, _, _, info = self.env.step(action)
                        o_2 = new_observation['observation']
                        ag_2 = new_observation['achieved_goal']
                        # append rollout
                        obs.append(o.copy())
                        achieved_goals.append(ag.copy())
                        goals.append(g.copy())
                        acts.append(action.copy())
                        # new vals for obs and ag
                        o = o_2
                        ag = ag_2
                    # append the last vals
                    obs.append(o.copy())
                    achieved_goals.append(ag.copy())
                    # record cycle
                    batch_obs.append(obs)
                    batch_ag.append(achieved_goals)
                    batch_g.append(goals)
                    batch_act.append(acts)
                # convert to array
                batch_obs = np.array(batch_obs)
                batch_ag = np.array(batch_ag)
                batch_g = np.array(batch_g)
                batch_act = np.array(batch_act)
                # store cycle
                self.buffer.store_episode([batch_obs, batch_ag, batch_g, batch_act])
                # update normalizer with sampled transitions
                self._update_normalizer([batch_obs, batch_ag, batch_g, batch_act])
                # part of mine
                for _ in range(self.n_update):
                    # train and update the model
                    self.update_models()
                self.soft_update(self.actor_target, self.actor)
                self.soft_update(self.critic_target, self.critic)
            # eval the agent
            success_rate = self._eval_agent()
            self.log['success'].append(success_rate)
            self.log['epoch'].append(epoch)
            self.summary()
            if MPI.COMM_WORLD.Get_rank() == 0:
                torch.save([self.o_norm.mean, self.o_norm.std, self.g_norm.mean, self.g_norm.std,
                            self.actor.state_dict(), self.log], self.model_path + '/model.pt')

        # plot_results(self.log['success'], self.log['mean_actor'], self.log['mean_critic'], self.env_params)

    def _preprocess_input(self, obs, g):
        # normalize obs and goal
        obs_norm = self.o_norm.normalize(obs)
        g_norm = self.g_norm.normalize(g)
        # concat
        input = np.concatenate([obs_norm, g_norm])
        input = torch.tensor(input, dtype=torch.float).unsqueeze(0)
        return input

    def get_action(self, o, g):
        input_tensor = self._preprocess_input(o, g)
        action = self.actor(input_tensor)
        action = action.cpu().numpy().squeeze()
        # print('action:', action)
        # add the gaussian
        action += self.noise_eps * self.env_params['action_max'] * np.random.randn(*action.shape)
        action = np.clip(action, -self.env_params['action_max'], self.env_params['action_max'])
        # random actions...
        random_actions = np.random.uniform(low=-self.env_params['action_max'], high=self.env_params['action_max'], \
                                           size=self.env_params['action'])
        # choose if use the random actions
        action += np.random.binomial(1, self.random_eps, 1)[0] * (random_actions - action)
        return action

    def _update_normalizer(self, episode_batch):
        obs, achieved_goals, goals, acts = episode_batch
        obs_next = obs[:, 1:, :]
        achieved_goals_next = achieved_goals[:, 1:, :]
        # get the number of normalization transitions
        num_transitions = acts.shape[1]
        # create the new buffer to store them
        buffer_temp = {'obs': obs,
                       'ag': achieved_goals,
                       'g': goals,
                       'actions': acts,
                       'obs_next': obs_next,
                       'ag_next': achieved_goals_next,
                       }
        transitions = self.HER.sample_her_transitions(buffer_temp, num_transitions)
        obs, g = transitions['obs'], transitions['g']
        # pre process the obs and g
        transitions['obs'], transitions['g'] = self._preprocess_og(obs, g)
        # update
        self.o_norm.update(transitions['obs'])
        self.g_norm.update(transitions['g'])
        # recompute the stats
        self.o_norm.recompute_stats()
        self.g_norm.recompute_stats()

    def _preprocess_og(self, o, g):
        o = np.clip(o, -self.clip_obs, self.clip_obs)
        g = np.clip(g, -self.clip_obs, self.clip_obs)
        return o, g

    # update the network
    def update_models(self):
        # sample episode online
        transitions = self.buffer.sample(self.batch_size)
        # preprocess the observations and goals
        obs, obs_next, goals = transitions['obs'], transitions['obs_next'], transitions['g']
        transitions['obs'], transitions['g'] = self._preprocess_og(obs, goals)
        transitions['obs_next'], transitions['g_next'] = self._preprocess_og(obs_next, goals)
        # start the update
        # norm obs and g
        obs_norm = self.o_norm.normalize(transitions['obs'])
        g_norm = self.g_norm.normalize(transitions['g'])
        inputs_norm = np.concatenate([obs_norm, g_norm], axis=1)
        # norm next obs and next g
        obs_next_norm = self.o_norm.normalize(transitions['obs_next'])
        g_next_norm = self.g_norm.normalize(transitions['g_next'])
        inputs_next_norm = np.concatenate([obs_next_norm, g_next_norm], axis=1)
        # transform all into tensor
        inputs_norm_t = torch.tensor(inputs_norm, dtype=torch.float)
        inputs_next_norm_t = torch.tensor(inputs_next_norm, dtype=torch.float)
        actions_t = torch.tensor(transitions['actions'], dtype=torch.float)
        rewards_t = torch.tensor(transitions['r'], dtype=torch.float)

        with torch.no_grad():
            next_actions = self.actor_target(inputs_next_norm_t)
            next_q_value = self.critic_target(inputs_next_norm_t, next_actions)
            next_q_value = next_q_value.detach()
            target_q = rewards_t + self.gamma * next_q_value
            target_q = target_q.detach()
            # clip the q value
            clip_return = 1 / (1 - self.gamma)
            target_q = torch.clamp(target_q, -clip_return, 0)       # clip target used to train critic as in the paper
        # the q loss
        real_q_vals = self.critic(inputs_norm_t, actions_t)
        critic_loss = torch.nn.MSELoss()(target_q, real_q_vals)     # (target_q - real_q_vals).pow(2).mean()
        # actor loss
        actions_real = self.actor(inputs_norm_t)
        actor_loss = -self.critic(inputs_norm_t, actions_real).mean()
        actor_loss += self.action_l2 * (actions_real / self.env_params['action_max']).pow(2).mean()  # L2 reg
        # update the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        sync_grads(self.actor)
        self.actor_optimizer.step()
        # update the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        sync_grads(self.critic)
        self.critic_optimizer.step()
        # record vals
        self.log['actor_loss'].append(actor_loss.detach().cpu().numpy())
        self.log['critic_loss'].append(critic_loss.detach().cpu().numpy())

    def soft_update(self, target_net, net):
        for target_param, param in zip(target_net.parameters(), net.parameters()):
            target_param.data.copy_(self.tau * target_param.data + (1 - self.tau) * param.data)

    def _eval_agent(self, render=False):
        total_success = []
        for _ in range(self.test_rollouts):
            current_success_rate = []
            observation = self.env.reset()
            o = observation['observation']
            g = observation['desired_goal']
            for _ in range(self.env_params['max_timesteps']):
                if render:
                    self.env.render()
                with torch.no_grad():
                    input_tensor = self._preprocess_input(o, g)
                    action = self.actor(input_tensor).cpu().numpy().squeeze()
                observation_new, _, _, info = self.env.step(action)
                o = observation_new['observation']
                g = observation_new['desired_goal']
                current_success_rate.append(info['is_success'])
            total_success.append(current_success_rate)
        total_success_rate = np.array(total_success)
        local_success_rate = np.mean(total_success_rate[:, -1])
        global_success_rate = MPI.COMM_WORLD.allreduce(local_success_rate, op=MPI.SUM)
        return global_success_rate / MPI.COMM_WORLD.Get_size()

    def summary(self):

        mean_actor = (np.mean(self.log["actor_loss"][-self.n_update:]))
        mean_critic = (np.mean(self.log["critic_loss"][-self.n_update:]))

        # print(f'-------------------------------------------------')
        print('[{}] | Epoch {}: {:.2f}     ||     Actor loss: {:.4f} | Critic loss: {:.4f}'
              .format(datetime.now(), self.log["epoch"][-1], self.log["success"][-1], mean_actor, mean_critic))

        self.log['mean_actor'].append(mean_actor)
        self.log['mean_critic'].append(mean_critic)


