from pysc2.env import sc2_env

def record(episode,
           episode_reward,
           worker_idx,
           global_ep_reward,
           result_queue,
           total_loss,
           num_steps):
    """Helper function to store score and print statistics.
    Arguments:
      episode: Current episode
      episode_reward: Reward accumulated over the current episode
      worker_idx: Which thread (worker)
      global_ep_reward: The moving average of the global reward
      result_queue: Queue storing the moving average of the scores
      total_loss: The total loss accumualted over the current episode
      num_steps: The number of steps the episode took to complete
    """
    if global_ep_reward == 0:
        global_ep_reward = episode_reward
    else:
        global_ep_reward = global_ep_reward * 0.99 + episode_reward * 0.01
    print(
        f"Episode: {episode} | "
        f"Moving Average Reward: {int(global_ep_reward)} | "
        f"Episode Reward: {int(episode_reward)} | "
        f"Loss: {int(total_loss / float(num_steps) * 1000) / 1000} | "
        f"Steps: {num_steps} | "
        f"Worker: {worker_idx}"
    )
    result_queue.put(global_ep_reward)
    return global_ep_reward

def generate_env(map_name, viz):
    """This function returns a new sc2 environment, with 64x64 dimensions.
    Parameters
    ----------
    map_name (String): the name of the sc2 minigame to generate.
    viz (Boolean): activate game visualization.

    Returns
    -------
    env (pysc2.env.sc2_env.SC2Env): the new environment."""
    env = sc2_env.SC2Env(agent_race=None,
                    bot_race=None,
                    difficulty=None,
                    map_name=map_name,
                    visualize=viz,agent_interface_format=sc2_env.AgentInterfaceFormat(
              feature_dimensions=sc2_env.Dimensions(
                  screen=64,
                  minimap=64)))

class Memory:
    """This class allows to keep track of states, actions and rewards at each step."""
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []

    def store(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []