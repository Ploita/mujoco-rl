from typing import Any
import gymnasium as gym

def create_env(flag: bool = True) -> tuple[Any, int, int]:
    """Creates and wraps the environment for training

    Returns:
        env: Wrapped environment
    """
    
    env = gym.make("CartPole-v1") #, render_mode = 'human')
    wrapped_env = gym.wrappers.RecordEpisodeStatistics(env, 50)
    obs_space_dims = env.observation_space.shape[0]
    action_space_dims = int(env.action_space.n)

    return wrapped_env, obs_space_dims, action_space_dims
