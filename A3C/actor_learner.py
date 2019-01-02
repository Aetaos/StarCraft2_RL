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
from keras.models import Sequential
from keras.layers import Dense,Conv1D,Dropout,Flatten,Activation,MaxPool1D,MaxPool2D
from keras.optimizers import Adam
from threading import Thread

class a3c_agent(Thread):
    """This class implements all useful methods and variables
    for an actor-critic thread, for the a3c learning algorithm."""

    def __init__(self, theta, theta_v, T):
        self.theta = theta
        self.theta_v = theta_v
        self.T = T


