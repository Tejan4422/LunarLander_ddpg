# -*- coding: utf-8 -*-
"""
Created on Fri May  7 11:07:35 2021

@author: Tejan
"""

import numpy as np

class ReplayBuffer(object):
    def __init__(self, max_size, input_shape, tot_act):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape))
        self.new_state_memory = np.zeros((self.mem_size, *input_shape))
        self.action_memory = np.zeros((self.mem_size, tot_act))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.float32)
    def save_transition(self, state, action, reward, new_state, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = new_state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = 1 - done
        self.mem_cntr += 1
    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size)
        state_memory = self.state_memory[batch]
        action_memory = self.action_memory[batch]
        reward_memory = self.reward_memory[batch]
        new_state_mem = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]
        return state_memory, action_memory, reward_memory, new_state_mem, terminal
