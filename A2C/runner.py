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

from A2C.a2c_agent import A2CAgent
from A2C.network import FullyConv, FullyConvLSTM
from A2C.utils import get_state, get_action

_AI_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_AI_SELECTED = features.SCREEN_FEATURES.selected.index
_NO_OP = actions.FUNCTIONS.no_op.id
_MOVE_SCREEN = actions.FUNCTIONS.Attack_screen.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_SELECT_POINT = actions.FUNCTIONS.select_point.id
_MOVE_RAND = 1000
_MOVE_MIDDLE = 2000
_BACKGROUND = 0
_AI_SELF = 1
_AI_ALLIES = 2
_AI_NEUTRAL = 3
_AI_HOSTILE = 4
_SELECT_ALL = [0]
_NOT_QUEUED = [0]
EPS_START = 0.9
EPS_END = 0.025
EPS_DECAY = 2500

# define our actions
# it can choose to move to
# the beacon or to do nothing
# it can select the marine or deselect
# the marine, it can move to a random point
categorical_actions = [
    _NO_OP,
    _SELECT_ARMY,
    #_SELECT_POINT,
    #_MOVE_RAND,
    #_MOVE_MIDDLE
]
spatial_actions=[    _MOVE_SCREEN,
]
id_from_actions={}
action_from_id={}
for ix,k in enumerate(spatial_actions):
    id_from_actions[k] = ix
    action_from_id[ix] = k
for ix,k in enumerate(categorical_actions):
    id_from_actions[k]=ix+len(spatial_actions)
    action_from_id[ix+len(spatial_actions)] = k


#initialize NN model hyperparameters
eta = 0.1
expl_rate = 0.1

#initialize model object
model = FullyConvLSTM(eta, expl_rate, categorical_actions,spatial_actions)

#initalize Agent
agent = A2CAgent(model, categorical_actions,spatial_actions, id_from_actions,action_from_id)

FLAGS = flags.FLAGS
FLAGS(['run_sc2'])

viz = False
save_replay = False
steps_per_episode = 0 # 0 actually means unlimited
MAX_EPISODES =100
MAX_STEPS = 400
steps = 0

# create a map
beacon_map = maps.get('MoveToBeacon')


#run trajectories and train
with sc2_env.SC2Env(agent_race=None,
                    bot_race=None,
                    difficulty=None,
                    map_name=beacon_map,
                    visualize=viz, agent_interface_format=sc2_env.AgentInterfaceFormat(
            feature_dimensions=sc2_env.Dimensions(
                screen=64,
                minimap=64))) as env:
    # agent.load("./save/move_2_beacon-dqn.h5")

    done = False
    # batch_size = 5

    for e in range(MAX_EPISODES):
        obs = env.reset()
        score = 0
        state = get_state(obs[0])
        for time in range(MAX_STEPS):
            # env.render()
            init = False
            if e == 0 and time == 0:
                init = True
            a,point = agent.act(state, init)
            if not a in obs[0].observation.available_actions:
                a = _NO_OP
            func = get_action(a, point)
            next_obs = env.step([func])
            next_state = get_state(next_obs[0])
            reward = float(next_obs[0].reward)
            score += reward
            done = next_obs[0].last()
            agent.append_sample(state, a, reward,point)
            state = next_state
            obs = next_obs
            if done:
                print("episode: {}/{}, score: {}"
                      .format(e, MAX_EPISODES, score))
                break
        agent.train()
        agent.save("./save/move_2_beacon-dqn.h5")