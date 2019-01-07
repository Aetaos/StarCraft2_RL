import numpy as np
from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features
from pysc2.env import sc2_env, run_loop, available_actions_printer
from pysc2 import maps

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

def get_state(obs):
    return [np.array(obs.observation['feature_screen']).reshape(1, 17, 64, 64),
            np.array(obs.observation['feature_minimap']).reshape(1, 7, 64, 64)]

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