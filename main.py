import gym
from ddpg_agent import DDPGAgent
from utils import *
# import all mujoco path and ddl authorizations
os_add_pathways()

# create env
env = gym.make('FetchReach-v1')
env_params = get_env_params(env)

ddpg = DDPGAgent(env, env_params)
ddpg.learn()
ddpg._eval_agent(render=True)


