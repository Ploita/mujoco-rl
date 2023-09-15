from typing import Any
import gymnasium as gym

def create_env() -> tuple[Any, int, int]:
    """Creates and wraps the environment for training

    Returns:
        env: Wrapped environment
    """
    
    env = gym.make("CartPole-v1") #, render_mode = 'human')
    wrapped_env = gym.wrappers.RecordEpisodeStatistics(env, 50) #type: ignore
    obs_space_dims = env.observation_space.shape[0] #type: ignore
    action_space_dims = int(env.action_space.n) #type: ignore

    return wrapped_env, obs_space_dims, action_space_dims
