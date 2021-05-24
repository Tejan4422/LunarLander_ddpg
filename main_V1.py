# -*- coding: utf-8 -*-
"""
Created on Fri May  7 13:06:53 2021

@author: Tejan
"""
from Agent import Agent
import gym
import numpy as np
from utils import plotLearning
from replay_buffer import ReplayBuffer
import pandas as pd

env = gym.make('LunarLanderContinuous-v2')


"""
agent = Agent(alpha=0.0002, beta=0.0002, input_dims=[8],
              tau=0.001, env=env,
              batch_size=32,  layer1_size=300, layer2_size=200, n_actions=2)

agent = Agent(alpha=0.001, beta=0.001, input_dims=[8], 
              tau=0.001, env=env,
              batch_size=32,  layer1_size=300, layer2_size=200, n_actions=2)

agent = Agent(lr=0.002, input_dims=[8], 
              tau=0.002, env=env,
              batch=32,  layer1_size=300, layer2_size=200, tot_actions=2)

"""
agent = Agent(lr=0.001, input_dims=[8], 
              tau=0.001, env=env,
              batch=32,  layer1_size=500, layer2_size=300, tot_actions=2)
#agent.load_models()
np.random.seed(0)

ep_score = []
mean_score = []

score_history = []
for i in range(1000):
    obs = env.reset()
    done = False
    score = 0
    while not done:
        act = agent.action_select(obs)
        new_state, reward, done, info = env.step(act)
        agent.remember(obs, act, reward, new_state, int(done))
        agent.learn()
        score += reward
        obs = new_state
        #env.render()
    score_history.append(score)

    #if i % 25 == 0:
    #    agent.save_models()
    ep_score.append(score)
    mean_score.append(np.mean(score_history[-100:]))
    print('episode ', i, 'score %.2f' % score,
          'trailing 100 games avg %.3f' % np.mean(score_history[-100:]))

df = pd.DataFrame(list(zip(ep_score, mean_score)), columns = ['Episode_score', 'mean_score'])
df.to_csv('default_para_alpha001_beta001_500_300_001.csv')

filename = 'LunarLander-alpha001-beta001-500-300_001.png'
plotLearning(score_history, filename, window=100)
