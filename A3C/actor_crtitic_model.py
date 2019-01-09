import keras

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

    def __init__(self, model, categorical_actions,spatial_actions, id_from_actions,action_from_id):
        self.gamma = 0.95  # discount rate
        self.categorical_actions = categorical_actions
        self.spatial_actions = spatial_actions
        self.model = model
        self.epsilon = 0.5
        self.id_from_actions = id_from_actions
        self.action_from_id = action_from_id

    def update_epsilon(self):
        if self.epsilon > 0.05:
            self.epsilon = 0.99 * self.epsilon

    
   
    def act(self, state, init=False):
        policy = (self.model.predict(state)[1]).flatten()
        
        if init or np.random.random() < self.epsilon:
            return self.action_from_id[np.random.choice(len(self.action_from_id), 1)[0]],np.random.randint(4096)
        else:
            preds=self.model.predict(state)
            return self.action_from_id[np.random.choice(len(self.action_from_id),1,p=preds[1][0])[0]],np.random.choice(4096,1,p=preds[2][0])[0]

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)
