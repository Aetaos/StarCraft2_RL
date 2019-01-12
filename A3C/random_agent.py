from pysc2 import maps
from pysc2.env import sc2_env
from queue import Queue
from .utils import record

class RandomAgent:
    """Random Agent that will play the specified game
      Arguments:
        max_eps: Maximum number of episodes to run agent for.
        map_name (String): the name of the mini_game map.
        viz (Boolean): activate game visualization.
    """
    def __init__(self, max_eps, map_name, viz):
        # generate sc2 environment
        self.env = sc2_env.SC2Env(agent_race=None,
                    bot_race=None,
                    difficulty=None,
                    map_name=maps.get(map_name),
                    visualize=viz,agent_interface_format=sc2_env.AgentInterfaceFormat(
              feature_dimensions=sc2_env.Dimensions(
                  screen=64,
                  minimap=64)))
        self.max_episodes = max_eps
        self.global_moving_average_reward = 0
        self.res_queue = Queue()

    def run(self):
        reward_avg = 0
        for episode in range(self.max_episodes):
            done = False
            self.env.reset()
            reward_sum = 0.0
            steps = 0
            while not done:
                # Sample randomly from the action space and step
                _, reward, done, _ = self.env.step(self.env.action_space.sample())
                steps += 1
                reward_sum += reward
            # Record statistics
            self.global_moving_average_reward = record(episode,
                                                       reward_sum,
                                                       0,
                                                       self.global_moving_average_reward,
                                                       self.res_queue, 0, steps)

            reward_avg += reward_sum
        final_avg = reward_avg / float(self.max_episodes)
        print("Average score across {} episodes: {}".format(self.max_episodes, final_avg))
        return final_avg
