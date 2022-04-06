import gym
from ddpg_agent import DDPGAgent
from utils import *
import random
import torch
import mpi4py.MPI as MPI
# import all mujoco path and ddl authorizations
os_add_pathways()

params_models = {
     'name_env': 'FetchReach-v1',
     'seed': 14,
     'fc_shape': 256,
     'actor_lr': 0.001,
     'critic_lr': 0.001,
     'replay_strategy': 'future',
     'replay_k': 4,
     'clip_range': 5,
     'clip_obs': 200,
     'memory_size': 100_00,
     'batch_size': 256,
     'n_epochs': 50,
     'n_cycles': 50,
     'rollout': 2,
     'n_update': 40,
     'test_rollouts': 10,
     'save_models': 5,
     'noise_eps': 0.2,
     'random_eps': 0.3,
     'gamma': 0.98,
     'tau': 0.95,
     'action_l2': 1
}

if __name__ == '__main__':

    env = gym.make(params_models['name_env'])
    env_params = get_env_params(env)

    env.seed(params_models['seed'] + MPI.COMM_WORLD.Get_rank())
    random.seed(params_models['seed'] + MPI.COMM_WORLD.Get_rank())
    np.random.seed(params_models['seed'] + MPI.COMM_WORLD.Get_rank())
    torch.manual_seed(params_models['seed'] + MPI.COMM_WORLD.Get_rank())

    # take the configuration for the HER
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['IN_MPI'] = '1'

    # train model
    ddpg_agent = DDPGAgent(env, env_params,
                           n_epochs=params_models['n_epochs'],
                           n_cycles=params_models['n_cycles'],
                           rollout=params_models['rollout'])
    ddpg_agent.learn()
    ddpg_agent._eval_agent(render=True)


