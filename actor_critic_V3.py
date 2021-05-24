# -*- coding: utf-8 -*-
"""
Created on Thu May  6 19:15:21 2021

@author: Tejan
"""
import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import gym
from replay_buffer import ReplayBuffer
env = gym.make('LunarLanderContinuous-v2')



class ActionNoise(object):
    def __init__(self, mu, sigma=0.15, theta=.2, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(
                                                            self.mu, self.sigma)
class CriticNetwork(nn.Module):
    def __init__(self, lr, inputp_dims, layer1, layer2, tot_actions, name):
        super(CriticNetwork, self).__init__()
        self.input_dims = inputp_dims
        self.layer1 = layer1
        self.layer2 = layer2
        self.tot_actions = tot_actions
        self.fc1 = nn.Linear(*self.input_dims, self.layer1)
        f1 = 1./np.sqrt(self.fc1.weight.data.size()[0])
        T.nn.init.uniform_(self.fc1.weight.data, -f1, f1)
        T.nn.init.uniform_(self.fc1.bias.data, -f1, f1)
        self.batch_norm1 = nn.LayerNorm(self.layer1)

        self.fc2 = nn.Linear(self.layer1, self.layer2)
        f2 = 1./np.sqrt(self.fc2.weight.data.size()[0])
        T.nn.init.uniform_(self.fc2.weight.data, -f2, f2)
        T.nn.init.uniform_(self.fc2.bias.data, -f2, f2)
        self.batch_norm2 = nn.LayerNorm(self.layer2)

        self.action_value = nn.Linear(self.tot_actions, self.layer2)
        f3 = 0.003
        self.q = nn.Linear(self.layer2, 1)
        T.nn.init.uniform_(self.q.weight.data, -f3, f3)
        T.nn.init.uniform_(self.q.bias.data, -f3, f3)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cuda:1')

        self.to(self.device)

    def critic_for(self, state, action):
        state = self.fc1(state)
        state = self.batch_norm1(state)
        state = F.relu(state)
        state = self.fc2(state)
        state = self.batch_norm2(state)

        action = F.relu(self.action_value(action))
        state_action_value = F.relu(T.add(state, action))
        state_action_value = self.q(state_action_value)

        return state_action_value


class ActorNetwork(nn.Module):
    def __init__(self, lr, input_dims, layer1, layer2, tot_actions, name):
        super(ActorNetwork, self).__init__()
        self.input_dims = input_dims
        self.layer1 = layer1
        self.layer2 = layer2
        self.tot_actions = tot_actions
        self.fc1 = nn.Linear(*self.input_dims, self.layer1)
        f1 = 1./np.sqrt(self.fc1.weight.data.size()[0])
        T.nn.init.uniform_(self.fc1.weight.data, -f1, f1)
        T.nn.init.uniform_(self.fc1.bias.data, -f1, f1)
        self.batch_norm1 = nn.LayerNorm(self.layer1)

        self.fc2 = nn.Linear(self.layer1, self.layer2)
        
        f2 = 1./np.sqrt(self.fc2.weight.data.size()[0])
        T.nn.init.uniform_(self.fc2.weight.data, -f2, f2)
        T.nn.init.uniform_(self.fc2.bias.data, -f2, f2)
        self.batch_norm2 = nn.LayerNorm(self.layer2)

        f3 = 0.003
        self.mu = nn.Linear(self.layer2, self.tot_actions)
        T.nn.init.uniform_(self.mu.weight.data, -f3, f3)
        T.nn.init.uniform_(self.mu.bias.data, -f3, f3)
 
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cuda:1')

        self.to(self.device)

    def actor_for(self, state):
        state = self.fc1(state)
        state = self.batch_norm1(state)
        state = F.relu(state)
        state = self.fc2(state)
        state = self.batch_norm2(state)
        state = F.relu(state)
        state = T.tanh(self.mu(state))

        return state


class Agent(object):
    def __init__(self, lr, input_dims, tau, env, gamma=0.99,
                 tot_actions=2, max_size=1000000, layer1_size=400,
                 layer2_size=300, batch=64):
        self.gamma = gamma
        self.tau = tau
        self.buffer = ReplayBuffer(max_size, input_dims, tot_actions)
        self.batch = batch

        self.actor = ActorNetwork(lr, input_dims, layer1_size,
                                  layer2_size, tot_actions = tot_actions,
                                  name='Actor')
        self.critic = CriticNetwork(lr, input_dims, layer1_size,
                                    layer2_size, tot_actions=tot_actions,
                                    name='Critic')

        self.target_actor = ActorNetwork(lr, input_dims, layer1_size,
                                         layer2_size, tot_actions=tot_actions,
                                         name='TargetActor')
        self.target_critic = CriticNetwork(lr, input_dims, layer1_size,
                                           layer2_size, tot_actions=tot_actions,
                                           name='TargetCritic')

        self.noise = ActionNoise(mu=np.zeros(tot_actions))

        self.update_network(tau=1)

    def action_select(self, obs):
        self.actor.eval()
        observation = T.tensor(obs, dtype=T.float).to(self.actor.device)
        mu = self.actor.actor_for(observation).to(self.actor.device)
        mu_prime = mu + T.tensor(self.noise(),
                                 dtype=T.float).to(self.actor.device)
        self.actor.train()
        return mu_prime.cpu().detach().numpy()


    def remember(self, state, action, reward, new_state, done):
        self.buffer.save_transition(state, action, reward, new_state, done)

    def learn(self):
        if self.buffer.mem_cntr < self.batch:
            return
        state, action, reward, new_state, done = \
                                      self.buffer.sample_buffer(self.batch)

        reward = T.tensor(reward, dtype=T.float).to(self.critic.device)
        done = T.tensor(done).to(self.critic.device)
        new_state = T.tensor(new_state, dtype=T.float).to(self.critic.device)
        action = T.tensor(action, dtype=T.float).to(self.critic.device)
        state = T.tensor(state, dtype=T.float).to(self.critic.device)

        self.target_actor.eval()
        self.target_critic.eval()
        self.critic.eval()
        target_act = self.target_actor.actor_for(new_state)
        critic_value_ = self.target_critic.critic_for(new_state, target_act)
        critic_value = self.critic.critic_for(state, action)

        target = []
        for j in range(self.batch):
            target.append(reward[j] + self.gamma*critic_value_[j]*done[j])
        target = T.tensor(target).to(self.critic.device)
        target = target.view(self.batch, 1)

        self.critic.train()
        self.critic.optimizer.zero_grad()
        critic_loss = F.mse_loss(target, critic_value)
        critic_loss.backward()
        self.critic.optimizer.step()

        self.critic.eval()
        self.actor.optimizer.zero_grad()
        mu = self.actor.actor_for(state)
        self.actor.train()
        actor_loss = -self.critic.critic_for(state, mu)
        actor_loss = T.mean(actor_loss)
        actor_loss.backward()
        self.actor.optimizer.step()

        self.update_network()

    def update_network(self, tau=None):
        if tau is None:
            tau = self.tau

        actor_params = self.actor.named_parameters()
        critic_params = self.critic.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        target_critic_params = self.target_critic.named_parameters()

        critic_parameters = dict(critic_params)
        actor_parameters = dict(actor_params)
        target_critic_parameters = dict(target_critic_params)
        target_actor_dict = dict(target_actor_params)

        for name in critic_parameters:
            critic_parameters[name] = tau*critic_parameters[name].clone() + \
                                      (1-tau)*target_critic_parameters[name].clone()

        self.target_critic.load_state_dict(critic_parameters)

        for name in actor_parameters:
            actor_parameters[name] = tau*actor_parameters[name].clone() + \
                                      (1-tau)*target_actor_dict[name].clone()
        self.target_actor.load_state_dict(actor_parameters)

