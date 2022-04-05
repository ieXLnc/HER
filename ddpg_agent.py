import os
from torch.optim import Adam
from networks import Actor, Critic
from utils import plot_results, os_add_pathways
from MPI.mpi_utils import *
from MPI.normalizer import normalizer
from buffers import ReplayBuffer, HERSample
from gym.wrappers.monitoring import video_recorder

os_add_pathways()


class DDPGAgent:
    def __init__(self,
                 env,
                 env_params,
                 fc_shape=64,
                 actor_lr=0.001,
                 critic_lr=0.001,
                 replay_strategy='future',
                 replay_k=4,
                 clip_range=5,
                 clip_obs=200,
                 memory_size=100_00,
                 batch_size=256,
                 n_epochs=2,
                 n_cycles=50,
                 rollout=2,
                 n_update=40,
                 test_rollouts=10,
                 noise_eps=0.2,
                 random_eps=0.3,
                 gamma=0.98,
                 tau=0.95,
                 action_l2=1
                 ):

        self.env = env
        self.env_params = env_params

        # hyperparams
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.replay_strategy = replay_strategy
        self.replay_k = replay_k

        self.clip_range = clip_range
        self.clip_obs = clip_obs
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.n_cycles = n_cycles
        self.rollout = rollout
        self.n_update = n_update
        self.test_rollouts = test_rollouts
        # noise
        self.noise_eps = noise_eps
        self.random_eps = random_eps

        self.gamma = gamma
        self.tau = tau
        self.action_l2 = action_l2

        # create the networks
        self.actor = Actor(env_params, fc_shape)
        self.critic = Critic(env_params, fc_shape)
        # sync them across cpu
        sync_networks(self.actor)
        sync_networks(self.critic)
        # build targets
        self.actor_target = Actor(env_params, fc_shape)
        self.critic_target = Critic(env_params, fc_shape)
        # hard update of the target networks
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        # create optim for the networks
        self.actor_optimizer = Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=self.critic_lr)
        # HER and ReplayBuffer
        self.HER = HERSample(self.replay_strategy, self.replay_k, self.env.compute_reward)
        self.buffer = ReplayBuffer(env_params, self.memory_size, self.HER.sample_her_transition)
        # Create normalizer
        self.obs_norm = normalizer(size=env_params['obs'], default_clip_range=self.clip_range)
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
            'actor_loss': [0],
            'critic_loss': [0],
            'epoch': [],
        }

    def learn(self):
        # train the algo
        for epoch in range(self.n_epochs):
            for _ in range(self.n_cycles):
                mb_obs, mb_ag, mb_g, mb_act = [], [], [], []
                for _ in range(self.rollout):
                    # reset rollout
                    ep_obs, ep_ag, ep_g, ep_act = [], [], [], []
                    # reset env
                    observation = self.env.reset()
                    obs = observation['observation']
                    ag = observation['achieved_goal']
                    g = observation['desired_goal']
                    # start collecting sample
                    for t in range(self.env_params['max_timesteps']):
                        with torch.no_grad():
                            input_obs = self._preprocess_input(obs, g)      # normalize and concat
                            action = self.get_action(input_obs)             # remove step because gaussian
                        # feed action
                        new_observation, _, _, info = self.env.step(action)
                        obs_new = new_observation['observation']
                        ag_new = new_observation['achieved_goal']
                        # append rollout
                        ep_obs.append(obs)
                        ep_ag.append(ag)
                        ep_g.append(g)
                        ep_act.append(action)
                        # new vals for obs and ag
                        obs = obs_new
                        ag = ag_new
                    # append the last vals
                    ep_obs.append(obs)
                    ep_ag.append(ag)
                    # record cycle
                    mb_obs.append(ep_obs)
                    mb_ag.append(ep_ag)
                    mb_g.append(ep_g)
                    mb_act.append(ep_act)
                # convert to array
                mb_obs = np.array(mb_obs)
                mb_ag = np.array(mb_ag)
                mb_g = np.array(mb_g)
                mb_act = np.array(mb_act)
                # store cycle
                self.buffer.store_episode([mb_obs, mb_ag, mb_g, mb_act])
                self._update_normalizer([mb_obs, mb_ag, mb_g, mb_act])
                for _ in range(self.n_update):
                    # train and update the model
                    self.update_models()
                self.soft_update(self.actor_target, self.actor)
                self.soft_update(self.critic_target, self.critic)
            # eval the agent
            success_rate = self._eval_agent()
            self.log['success'].append(success_rate)
            self.log['epoch'].append(epoch)
            if MPI.COMM_WORLD.Get_rank() == 0:
                self.summary()
                torch.save([self.obs_norm.mean, self.obs_norm.std, self.g_norm.mean, self.g_norm.std,
                            self.actor.state_dict()], self.model_path + '/model.pt')

        plot_results(self.log['success'], self.log['actor_loss'], self.log['critic_loss'], self.env_params)

    def _preprocess_input(self, obs, g):
        # normalize obs and goal
        obs_norm = self.obs_norm.normalize(obs)
        g_norm = self.g_norm.normalize(g)
        # concat
        input = np.concatenate([obs_norm, g_norm])
        input = torch.tensor(input, dtype=torch.float).unsqueeze(0)
        return input

    def get_action(self, obs_norm_concat):
        action = self.actor(obs_norm_concat)
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
        mb_obs, mb_ag, mb_g, mb_actions = episode_batch
        mb_obs_next = mb_obs[:, 1:, :]
        mb_ag_next = mb_ag[:, 1:, :]
        # get the number of normalization transitions
        num_transitions = mb_actions.shape[1]
        # create the new buffer to store them
        buffer_temp = {'obs': mb_obs,
                       'ag': mb_ag,
                       'g': mb_g,
                       'actions': mb_actions,
                       'obs_next': mb_obs_next,
                       'ag_next': mb_ag_next,
                       }
        transitions = self.HER.sample_her_transition(buffer_temp, num_transitions)
        obs, g = transitions['obs'], transitions['g']
        # pre process the obs and g
        transitions['obs'], transitions['g'] = self._preproc_og(obs, g)
        # update
        self.obs_norm.update(transitions['obs'])
        self.g_norm.update(transitions['g'])
        # recompute the stats
        self.obs_norm.recompute_stats()
        self.g_norm.recompute_stats()

    def _preproc_og(self, o, g):
        o = np.clip(o, -self.clip_obs, self.clip_obs)
        g = np.clip(g, -self.clip_obs, self.clip_obs)
        return o, g

    def update_models(self):
        # sample episode online
        transitions = self.buffer.sample(self.batch_size)
        # preprocess the observations and goals
        obs, obs_next, goals = transitions['obs'], transitions['obs_next'], transitions['g']
        transitions['obs'], transitions['g'] = self._preproc_og(obs, goals)
        transitions['obs_next'], transitions['g_next'] = self._preproc_og(obs_next, goals)
        # start the update
        # norm obs and g
        obs_norm = self.obs_norm.normalize(transitions['obs'])
        g_norm = self.g_norm.normalize(transitions['g'])
        inputs_norm = np.concatenate([obs_norm, g_norm], axis=1)
        # norm next obs and next g
        obs_next_norm = self.obs_norm.normalize(transitions['obs_next'])
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
            target_q = torch.clamp(target_q, -clip_return, 0)  # clip target used to train critic as in the paper
        # the q loss
        real_q_vals = self.critic(inputs_norm_t, actions_t)
        critic_loss = torch.nn.MSELoss()(target_q, real_q_vals)
        # self.log['critic_loss'].append(critic_loss.detach().cpu().numpy())
        # actor loss
        actions_real = self.actor(inputs_norm_t)
        actor_loss = - self.critic.forward(inputs_norm_t, actions_real).mean()
        actor_loss += self.action_l2 * (actions_real / self.env_params['action_max']).pow(2).mean()  # L2 reg
        # record vals
        self.log['actor_loss'].append(actor_loss.detach().cpu().numpy())
        self.log['critic_loss'].append(critic_loss.detach().cpu().numpy())
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

    def soft_update(self, target_net, net):
        for target_params, params in zip(target_net.parameters(), net.parameters()):
            target_params.data.copy_(self.tau * params.data + (1.0 - self.tau) * target_params.data)

    def _eval_agent(self, render=False):
        total_success = []
        for _ in range(self.test_rollouts):
            current_success_rate = []
            observation = self.env.reset()
            obs = observation['observation']
            g = observation['desired_goal']
            for _ in range(self.env_params['max_timesteps']):
                if render:
                    self.env.render()
                with torch.no_grad():
                    input_tensor = self._preprocess_input(obs, g)
                    action = self.actor(input_tensor).cpu().numpy().squeeze()
                observation_new, _, _, info = self.env.step(action)
                obs = observation_new['observation']
                g = observation_new['desired_goal']
                current_success_rate.append(info['is_success'])
            total_success.append(current_success_rate)
        total_success_rate = np.array(total_success)
        local_success_rate = np.mean(total_success_rate[:, -1])
        global_success_rate = MPI.COMM_WORLD.allreduce(local_success_rate, op=MPI.SUM)
        return global_success_rate / MPI.COMM_WORLD.Get_size()

    def summary(self):
        print(f'-------------------------------------------------')
        print(f'Mean success for epoch {self.log["epoch"][-1]}: {self.log["success"][-1]}')
        print(f'Actor loss: {self.log["actor_loss"][-1]} | Critic loss: {self.log["critic_loss"][-1]}')


