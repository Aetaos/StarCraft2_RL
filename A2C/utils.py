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
"""transform a scalar from [0;4096] to a (y,x) coordinate in 64,64"""
def to_yx(point):
    return point%64, (point-(point%64))/64
def get_action(id_action,point):
    
    y,x = to_yx(point)
    if id_action== _NO_OP:
        func = actions.FunctionCall(_NO_OP, [])
    elif id_action == _MOVE_SCREEN:
        func = actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, [y, x]])
    elif id_action == _SELECT_ARMY:
        func = actions.FunctionCall(_SELECT_ARMY, [_SELECT_ALL])
    return func