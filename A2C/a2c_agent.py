import math
import numpy as np
from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features
from pysc2.env import sc2_env, run_loop, available_actions_printer
from pysc2 import maps
from absl import flags
from collections import deque
import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense,Conv1D,Conv2D,Dropout,Flatten,Activation,MaxPool1D,MaxPooling2D,Lambda
from keras.optimizers import Adam, RMSprop

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))



class A2CAgent:
    """This class implements the A2C agent using the network model"""

    def __init__(self, model, possible_actions, id_from_actions):
        self.states = []
        self.rewards = []
        self.actions = []
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # discount rate
        self.possible_actions = possible_actions
        self.initial_policy = [1 / len(possible_actions) for k in possible_actions]
        self.model = model
        self.epsilon = 0.5
        self.id_from_actions = id_from_actions

    def update_epsilon(self):
        if self.epsilon > 0.05:
            self.epsilon = 0.95 * self.epsilon

    def append_sample(self, state, action, reward):
        self.states.append(state)
        self.rewards.append(reward)
        self.actions.append(self.id_from_actions[action])

    def discount_rewards(self, rewards):
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(0, len(rewards))):
            running_add = running_add * self.gamma + rewards[t]
            discounted_rewards[t] = running_add
        return discounted_rewards

    def act(self, state, init=False):
        policy = (self.model.predict(state)[1]).flatten()
        if init or np.random.random() < self.epsilon:
            return self.possible_actions[np.random.choice(len(self.possible_actions), 1)[0]]
        else:
            return self.possible_actions[np.random.choice(len(self.possible_actions), 1, p=policy)[0]]

    def train(self):
        episode_length = len(self.states)
        discounted_rewards = self.discount_rewards(self.rewards)
        # Standardized discounted rewards
        """discounted_rewards -= np.mean(discounted_rewards) 
        if np.std(discounted_rewards):
            discounted_rewards /= np.std(discounted_rewards)
        else:
            self.states, self.actions, self.rewards = [], [], []
            #print ('std = 0!')
            return 0"""

        update_inputs = [np.zeros((episode_length, 17, 64, 64)),
                         np.zeros((episode_length, 7, 64, 64))]  # Episode_lengthx64x64x4

        # Episode length is like the minibatch size in DQN
        for i in range(episode_length):
            update_inputs[0][i, :, :, :] = self.states[i][0][0, :, :, :]
            update_inputs[1][i, :, :, :] = self.states[i][1][0, :, :, :]

        values = self.model.predict(update_inputs)[0]

        advantages = np.zeros((episode_length, len(self.possible_actions)))

        for i in range(episode_length):
            advantages[i][self.actions[i]] = discounted_rewards[i] - values[i]

        self.model.fit(update_inputs, [discounted_rewards, advantages], nb_epoch=1, verbose=0)

        self.states, self.actions, self.rewards = [], [], []

        self.update_epsilon()

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)