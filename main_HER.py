import os

import torch
import numpy as np
os.environ['PATH'] += r";C:\Users\xavier\.mujoco\mjpro150\bin"
os.add_dll_directory("C://Users//xavier//.mujoco//mjpro150//bin")
os.environ['PATH'] += r";C:\Users\xavier\.ffmpeg\ffmpeg-2022-02-17-git-2812508086-essentials_build\bin"

import mujoco_py
import gym
import gym_robotics
from utils import *
from ddpg_HER import DDPGAgent


env = gym.make('FetchReach-v1')
env_params = get_env_params(env)    # get params
print('env_params:', env_params)
env = HERGoalEnvWrapper(env)        # wrap env

Agent = DDPGAgent(env, env_params)
Agent.train()
Agent.eval_agent(render=True)

