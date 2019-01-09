import os
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
import tensorflow as tf
import keras 

from queue import Queue
from worker import Worker
from utils import generate_env
from actor_crtitic_model import A2CAgent
from network import FullyConv

#to do generate a proper env in a separate file
from pysc2.lib import actions
from pysc2.lib import features
from pysc2.env import sc2_env, run_loop, available_actions_printer
from pysc2 import maps
from absl import flags
_NO_OP = actions.FUNCTIONS.no_op.id
_MOVE_SCREEN = actions.FUNCTIONS.Attack_screen.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_SELECT_POINT = actions.FUNCTIONS.select_point.id


import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

#from A2C.a2c_agent import A2CAgent
#from A2C.network import FullyConv
#from A2C.utils import get_state, get_action

class MasterAgent():
    """This class optimizes the global parameter network, by launching several actor-critic agents
    on independent environments."""
    def __init__(self):#, args):
        #self.args = args
        save_dir = "./save"#self.args.save_dir
        self.save_dir = save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

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
        #with sc2_env.SC2Env(agent_race=None,
        """                bot_race=None,
                        difficulty=None,
                        map_name=beacon_map,
                        visualize=viz, agent_interface_format=sc2_env.AgentInterfaceFormat(
                feature_dimensions=sc2_env.Dimensions(
                    screen=64,
                    minimap=64))) as env:
""" 

        self.env = generate_env(beacon_map)
        #TODO adapt state and action sizes to pysc2 env
        #self.state_size = env.observation_space.shape[0]
        #self.action_size = env.action_space.n

        self.opt = tf.train.RMSPropOptimizer(0.01, decay=0.99, epsilon=1e-10, use_locking=True)
        #print(self.state_size, self.action_size)

        #self.global_model = ActorCriticModel(self.state_size, self.action_size)  # global network
        #self.gobal_model = A2CAgent()
        self.categorical_actions = [
            _NO_OP,
            _SELECT_ARMY,
            #_SELECT_POINT,
            #_MOVE_RAND,
            #_MOVE_MIDDLE
        ]
        self.spatial_actions=[    _MOVE_SCREEN,
        ]
        self.id_from_actions={}
        self.action_from_id={}
        for ix,k in enumerate(self.spatial_actions):
            self.id_from_actions[k] = ix
            self.action_from_id[ix] = k
        for ix,k in enumerate(self.categorical_actions):
            self.id_from_actions[k]=ix+len(self.spatial_actions)
            self.action_from_id[ix+len(self.spatial_actions)] = k
        
        
        #initialize NN model hyperparameters
        self.eta = 0.1
        self.expl_rate = 0.2
        
        #initialize model object
        self.global_model = FullyConv(self.eta, self.expl_rate, self.categorical_actions,self.spatial_actions)
        
        #initalize Agent
        self.agent = A2CAgent(self.global_model, self.categorical_actions,self.spatial_actions, self.id_from_actions,self.action_from_id)

        
        #self.global_model(tf.convert_to_tensor(np.random.random((1, self.state_size)), dtype=tf.float32))

    def train(self):
        #TODO replace with pysc2 random agent
        try:
            
        
            res_queue = Queue()
    
            workers = [Worker(self.categorical_actions,
                 self.spatial_actions,
                 self.global_model,
                 self.opt,
                 res_queue,
                 i,
                 ) for i in range(1)]
            #in range(multiprocessing.cpu_count())]
            print ("running on", multiprocessing.cpu_count(), "core on Romain 's war machine")
    
            for i, worker in enumerate(workers):
                print("Starting worker {}".format(i))
                worker.start()
    
            moving_average_rewards = []  # record episode reward to plot
            while True:
                reward = res_queue.get()
                if reward is not None:
                    moving_average_rewards.append(reward)
                else:
                    break
            [w.join() for w in workers]
    
            plt.plot(moving_average_rewards)
            plt.ylabel('Moving average ep reward')
            plt.xlabel('Step')
            plt.savefig(os.path.join(self.save_dir,
                                     '{} Moving Average.png'.format(self.game_name)))
            plt.show()
        finally :
            self.env.close()

    def play(self):
        return 0
        #env = generate_env(self.game_name)
        #state = env.reset()
        #model = self.global_model
        #model_path = os.path.join(self.save_dir, 'model_{}.h5'.format(self.game_name))
        #print('Loading model from: {}'.format(model_path))
        #model.load_weights(model_path)
        #done = False
        #step_counter = 0
        #reward_sum = 0

        #try:
         #   while not done:
          #      env.render(mode='rgb_array')
           #     policy, value = model(tf.convert_to_tensor(state[None, :], dtype=tf.float32))
            #    policy = tf.nn.softmax(policy)
             #   action = np.argmax(policy)
              #  state, reward, done, _ = env.step(action)
               # reward_sum += reward
                #print("{}. Reward: {}, action: {}".format(step_counter, reward_sum, action))
                #step_counter += 1
        #except KeyboardInterrupt:
         #   print("Received Keyboard Interrupt. Shutting down.")
        #finally:
         #   env.close()
if __name__ == "__main__":
    master = MasterAgent()
    master.train()
    