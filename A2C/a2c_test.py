# coding: utf-8
### A2C code exported from the IPYNB notebook with the same name

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
from keras.layers import Dense,Conv1D,Conv2D,Dropout,Flatten,Activation,MaxPool1D,MaxPooling2D
from keras.optimizers import Adam, RMSprop


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
#it can choose to move to the beacon or to do nothing 
#it can select the marine or deselect the marine, it can move to a random point
possible_actions = [
    _NO_OP,
    _SELECT_ARMY,
    _SELECT_POINT,
    _MOVE_SCREEN,
    _MOVE_RAND,
    _MOVE_MIDDLE
]
id_from_actions={}
for ix,k in enumerate(possible_actions):
    id_from_actions[k]=ix

def get_state(obs):
    #ai_view = obs.observation['feature_screen'][_AI_RELATIVE]
    #beaconxs, beaconys = (ai_view == _AI_NEUTRAL).nonzero()
    #marinexs, marineys = (ai_view == _AI_SELF).nonzero()
    #marinex, mariney = marinexs.mean(), marineys.mean()
        
    #marine_on_beacon = np.min(beaconxs) <= marinex <=  np.max(beaconxs) and np.min(beaconys) <= mariney <=  np.max(beaconys)
        
    # get a 1 or 0 for whether or not our marine is selected
    #ai_selected = obs.observation['feature_screen'][_AI_SELECTED]
    #marine_selected = int((ai_selected == 1).any())
    #return [np.array([ai_view]),np.array([marine_selected])]
    return [np.array(obs.observation['feature_screen']).reshape(1,17,64,64), np.array(obs.observation['feature_minimap']).reshape(1,7,64,64)]


# ### Fullyconv LSTM agent
# TODO: maybe change padding from valid to same

#map conv
input_map = keras.layers.Input(shape=(17,64,64),name='input_map')
model_view_map = Conv2D(16, kernel_size=(5,5), data_format='channels_first', input_shape=(17,64,64), kernel_initializer="he_uniform")(input_map)
model_view_map = Activation('relu')(model_view_map)
model_view_map = MaxPooling2D(pool_size=(2,2), strides=None, padding='valid', data_format='channels_first')(model_view_map)
model_view_map = Conv2D(32, kernel_size=(3,3), data_format='channels_first', kernel_initializer="he_uniform")(model_view_map)
model_view_map = Activation('relu')(model_view_map)
model_view_map = MaxPooling2D(pool_size=(2,2), strides=None, padding='valid', data_format='channels_first')(model_view_map)


#minimap conv
input_mini = keras.layers.Input(shape=(7,64,64),name='input_mini')
model_view_mini = Conv2D(16, kernel_size=(5,5), data_format='channels_first', input_shape=(7,64,64), kernel_initializer="he_uniform")(input_mini)
model_view_mini = Activation('relu')(model_view_mini)
model_view_mini = MaxPooling2D(pool_size=(2,2), strides=None, padding='valid', data_format='channels_first')(model_view_mini)
model_view_mini = Conv2D(32, kernel_size=(3,3), data_format='channels_first', kernel_initializer="he_uniform")(model_view_mini)
model_view_mini = Activation('relu')(model_view_mini)
model_view_mini = MaxPooling2D(pool_size=(2,2), strides=None, padding='valid', data_format='channels_first')(model_view_mini)

#non-spatial features
#input_feat = keras.layers.Input(shape=(1,))
# equivalent to added = keras.layers.add([x1, x2])

#concatenate
added = keras.layers.concatenate([model_view_map, model_view_mini])

#LSTM

added = Flatten()(added)
intermediate = keras.layers.Dense(256,activation='relu', kernel_initializer="he_uniform")(added)
out_value = keras.layers.Dense(1)(intermediate)
out_value = Activation('linear',name='value_output')(out_value)
out_non_spatial = keras.layers.Dense(len(possible_actions), kernel_initializer="he_uniform")(intermediate)
out_non_spatial = Activation('softmax', name='non_spatial_output')(out_non_spatial)
model = keras.models.Model(inputs=[input_map, input_mini], outputs=[out_value, out_non_spatial])
#model.summary()
losses={
    "value_output":"mse",
    "non_spatial_output":"categorical_crossentropy"
}
lossWeights = {"value_output": 1.0, "non_spatial_output": 1.0}
model.compile(loss=losses, loss_weights=lossWeights, optimizer=RMSprop(lr=0.1))


EPISODES = 500
import random


class A2CAgent:
    def __init__(self, model):
        self.states = []
        self.rewards = []
        self.actions = []
        
        self.gamma = 0.95    # discount rate
        self.epsilon= 0.5
        self.model = model

    def append_sample(self, state, action, reward):
        self.states.append(state)
        self.rewards.append(reward)
        self.actions.append(id_from_actions[action])
    
    def discount_rewards(self, rewards):
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(0, len(rewards))):
            if rewards[t] != 0:
                running_add = 0
            running_add = running_add * self.gamma + rewards[t]
            discounted_rewards[t] = running_add
        return discounted_rewards
    def update_epsilon(self):
        if self.epsilon>0.005:
            self.epsilon *= 0.95
    def act(self, state, init=False):
        policy = (self.model.predict(state)[1]).flatten()
        if  init or np.random.random()< self.epsilon:
            return possible_actions[np.random.choice(len(possible_actions),1)[0]]
        else:
          
            return possible_actions[np.random.choice(len(possible_actions),1,p=policy)[0]]

        
        
    def train(self):
        self.update_epsilon()
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
        
        update_inputs = [np.zeros((episode_length,17,64,64)),np.zeros((episode_length,7,64,64)) ] # Episode_lengthx64x64x4

        # Episode length is like the minibatch size in DQN
        for i in range(episode_length):
            update_inputs[0][i,:,:,:]=self.states[i][0][0,:,:,:]
            update_inputs[1][i,:,:,:]=self.states[i][1][0,:,:,:]
            
            
        values = self.model.predict(update_inputs)[0]
        
        advantages = np.zeros((episode_length, len(possible_actions)))

        for i in range(episode_length):
            advantages[i][self.actions[i]] = discounted_rewards[i] - values[i]
            
        self.model.fit(update_inputs, [discounted_rewards,advantages], nb_epoch=1, verbose=0)
        
        self.states, self.actions, self.rewards = [], [], []

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

FLAGS = flags.FLAGS
FLAGS(['run_sc2'])

viz = False
save_replay = False
steps_per_episode = 0 # 0 actually means unlimited
MAX_EPISODES = 200
MAX_STEPS = 400
steps = 0

# create a map
beacon_map = maps.get('MoveToBeacon')


def get_action(id_action,feature_screen):
    beacon_pos = (feature_screen == _AI_NEUTRAL).nonzero()

    if id_action== _NO_OP:
        func = actions.FunctionCall(_NO_OP, [])
    elif id_action == _MOVE_SCREEN:
        beacon_x, beacon_y = beacon_pos[0].mean(), beacon_pos[1].mean()
        func = actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, [beacon_y, beacon_x]])
    elif id_action == _SELECT_ARMY:
        func = actions.FunctionCall(_SELECT_ARMY, [_SELECT_ALL])
    elif id_action == _SELECT_POINT:
        backgroundxs, backgroundys = (feature_screen == _BACKGROUND).nonzero()
        point = np.random.randint(0, len(backgroundxs))
        backgroundx, backgroundy = backgroundxs[point], backgroundys[point]
        func = actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED, [backgroundy, backgroundx]])
    elif id_action == _MOVE_RAND:
        beacon_x, beacon_y = beacon_pos[0].max(), beacon_pos[1].max()
        movex, movey = np.random.randint(beacon_x, 64), np.random.randint(beacon_y, 64)
        func = actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, [movey, movex]])
    elif id_action == _MOVE_MIDDLE:
        func = actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, [32, 32]])
    return func


with sc2_env.SC2Env(agent_race=None,
                    bot_race=None,
                    difficulty=None,
                    map_name=beacon_map,
                    visualize=viz,
                    agent_interface_format=sc2_env.AgentInterfaceFormat(
              feature_dimensions=sc2_env.Dimensions(
                  screen=64,
                  minimap=64))) as env :
    agent = A2CAgent(model)
    #agent.load("./save/move_2_beacon-dqn.h5")
    
    done = False
    #batch_size = 32
    
    for e in range(MAX_EPISODES):
        obs = env.reset()
        score=0
        state = get_state(obs[0])
        for time in range(MAX_STEPS):
            #env.render()
            init=False
            if e==0 and time==0:
                init=True
            a=agent.act(state, init)
            print(a)
            if not a in obs[0].observation.available_actions:
                a=_NO_OP
            func=get_action(a,state[0][0][_AI_RELATIVE])
            next_obs=env.step([func])
            next_state = get_state(next_obs[0])
            reward = float(next_obs[0].reward)
            score+= reward
            done=next_obs[0].last()
            agent.append_sample(state,a,reward)
            state = next_state
            obs=next_obs
            if done:
                print("episode: {}/{}, score: {}"
                      .format(e, EPISODES, score))
                break
            #if len(agent.states) > batch_size:
             #   agent.train()
        agent.train()
        agent.save("./save/move_2_beacon-dqn.h5")
