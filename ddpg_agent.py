import os
from torch.optim import Adam
from networks import Actor, Critic
from utils import plot_results
from MPI.mpi_utils import *
from MPI.normalizer import normalizer
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
                 fc_shape=256,
                 actor_lr=0.001,
                 critic_lr=0.001,
                 replay_strategy='future',
                 replay_k=4,
                 clip_range=5,
                 clip_obs=200,
                 memory_size=100_00,
                 batch_size=256,
                 n_epochs=100,
                 n_cycles=50,
                 rollout=4,
                 n_update=40,
                 test_rollouts=10,
                 save_models=5,
                 noise_eps=0.2,
                 random_eps=0.3,
                 gamma=0.98,
                 tau=0.95,
                 action_l2=1
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
        self.save_models = save_models
        # noise
        self.noise_eps = noise_eps
        self.random_eps = random_eps

        self.gamma = gamma
        self.tau = tau
        self.action_l2 = action_l2

        # create the networks
        self.actor = Actor(env_params, fc_shape).to(device)
        self.critic = Critic(env_params, fc_shape).to(device)
        # sync them across cpu
        sync_networks(self.actor)
        sync_networks(self.critic)
        # build targets
        self.actor_target = Actor(env_params, fc_shape).to(device).to(device)
        self.critic_target = Critic(env_params, fc_shape).to(device).to(device)
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
                        obs.append(o)
                        achieved_goals.append(ag)
                        goals.append(g)
                        acts.append(action)
                        # new vals for obs and ag
                        o = o_2
                        ag = ag_2
                    # append the last vals
                    obs.append(o)
                    achieved_goals.append(ag)
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
                torch.save([self.obs_norm.mean, self.obs_norm.std, self.g_norm.mean, self.g_norm.std,
                            self.actor.state_dict(), self.log], self.model_path + '/model.pt')

        plot_results(self.log['success'], self.log['mean_actor'], self.log['mean_critic'], self.env_params)

    def _preprocess_input(self, obs, g):
        # normalize obs and goal
        obs_norm = self.obs_norm.normalize(obs)
        g_norm = self.g_norm.normalize(g)
        # concat
        input = np.concatenate([obs_norm, g_norm])
        input = torch.tensor(input, dtype=torch.float).unsqueeze(0).to(device)
        return input

    def get_action(self, o, g):
        input_obs = self._preprocess_input(o, g)
        action = self.actor(input_obs)
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
        transitions = self.HER.sample_her_transition(buffer_temp, num_transitions)
        obs, g = transitions['obs'], transitions['g']
        # pre process the obs and g
        transitions['obs'], transitions['g'] = self._preprocess_og(obs, g)
        # update
        self.obs_norm.update(transitions['obs'])
        self.g_norm.update(transitions['g'])
        # recompute the stats
        self.obs_norm.recompute_stats()
        self.g_norm.recompute_stats()

    def _preprocess_og(self, o, g):
        o = np.clip(o, -self.clip_obs, self.clip_obs)
        g = np.clip(g, -self.clip_obs, self.clip_obs)
        return o, g

    def update_models(self):
        # sample episode online
        transitions = self.buffer.sample(self.batch_size)
        # preprocess the observations and goals
        obs, obs_next, goals = transitions['obs'], transitions['obs_next'], transitions['g']
        transitions['obs'], transitions['g'] = self._preprocess_og(obs, goals)
        transitions['obs_next'], transitions['g_next'] = self._preprocess_og(obs_next, goals)
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
        inputs_norm_t = torch.tensor(inputs_norm, dtype=torch.float).to(device)
        inputs_next_norm_t = torch.tensor(inputs_next_norm, dtype=torch.float).to(device)
        actions_t = torch.tensor(transitions['actions'], dtype=torch.float).to(device)
        rewards_t = torch.tensor(transitions['r'], dtype=torch.float).to(device)

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
            target_params.data.copy_(self.tau * target_params.data + (1.0 - self.tau) * params.data)

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

        print(f'-------------------------------------------------')
        print('[{}] | Epoch {}: {:.2f}     ||     Actor loss: {:.4f} | Critic loss: {:.4f}'.format(datetime.now(), self.log["epoch"][-1], self.log["success"][-1], mean_actor, mean_critic))

        self.log['mean_actor'].append(mean_actor)
        self.log['mean_critic'].append(mean_critic)


