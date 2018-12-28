import numpy as np
from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features
from pysc2.env import sc2_env, run_loop, available_actions_printer
from pysc2 import maps
from absl import flags

# define the features the AI can seee
_AI_RELATIVE = features.SCREEN_FEATURES.player_relative.index
# define contstants for actions
_NO_OP = actions.FUNCTIONS.no_op.id
_MOVE_SCREEN = actions.FUNCTIONS.Attack_screen.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
# define constants about AI's world
_BACKGROUND = 0
_AI_SELF = 1
_AI_ALLIES = 2
_AI_NEUTRAL = 3
_AI_HOSTILE = 4
# constants for actions
_SELECT_ALL = [0]
_NOT_QUEUED = [0]

def get_marine_location(ai_relative_view):
    '''get the indices where the world is equal to 1'''
    return (ai_relative_view == _AI_SELF).nonzero()
    
def get_rand_location(ai_location):
    '''gets a random location at least n away from current x,y point.'''
    return [np.random.randint(0, 64), np.random.randint(0, 64)]
    
class Agent1(base_agent.BaseAgent):
    """An agent for doing a simple movement form one point to another."""
    def step(self, obs):
        '''step function gets called automatically by pysc2 environment'''
        # call the parent class to have pysc2 setup rewards/etc for us
        super(Agent1, self).step(obs)
        # if we can move our army (we have something selected)
        if _MOVE_SCREEN in obs.observation['available_actions']:
            # get what the ai can see about the world
            ai_view = obs.observation['feature_screen'][_AI_RELATIVE]
            # get the location of our marine in this world
            marine_x, marine_y = get_marine_location(ai_view)
            # it our marine is not on the screen do nothing.
            # this happens if we scroll away and look at a different
            # part of the world
            if not marine_x.any():
                return actions.FunctionCall(_NO_OP, [])
            target = get_rand_location([marine_x, marine_y])
            return actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, target])
        # if we can't move, we havent selected our army, so selecto ur army
        else:
            return actions.FunctionCall(_SELECT_ARMY, [_SELECT_ALL])
def get_beacon_location(ai_relative_view):
    '''returns the location indices of the beacon on the map'''
    return (ai_relative_view == _AI_NEUTRAL).nonzero() 
    
class Agent2(base_agent.BaseAgent):
    """An agent for doing a simple movement form one point to another."""
    def step(self, obs):
        '''Step function gets called automatically by pysc2 environment'''
        super(Agent2, self).step(obs)
        if _MOVE_SCREEN in obs.observation['available_actions']:
            ai_view = obs.observation['feature_screen'][_AI_RELATIVE]
            # get the beacon coordinates
            beacon_xs, beacon_ys = get_beacon_location(ai_view)
            if not beacon_ys.any():
                return actions.FunctionCall(_NO_OP, [])
            # get the middle of the beacon and move there
            target = [beacon_ys.mean(), beacon_xs.mean()]
            return actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, target])
        else:
            return actions.FunctionCall(_SELECT_ARMY, [_SELECT_ALL])