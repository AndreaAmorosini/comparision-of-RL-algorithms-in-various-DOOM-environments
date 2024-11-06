import gymnasium as gym
from gymnasium import Wrapper
from vizdoom import gymnasium_wrapper
from collections import deque
import numpy as np

class CustomVizDoomWrapper(Wrapper):
    def __init__(self, env, normalize=False, stack_frames = False, stack_size = 4):
        super().__init__(env)
        
        self.stack_frames = stack_frames
        self.normalize = normalize
        self.stack_size = stack_size
        self.frames = deque(maxlen=stack_size)
        
        image_space = env.observation_space["screen"]
        game_var_space = env.observation_space["gamevariables"]
        
        stacked_image_shape = (stack_size, *image_space.shape)
        self.observation_space = gym.spaces.Dict({
            "frame" : gym.spaces.Box(low=0, high=255, shape=stacked_image_shape, dtype=np.uint8),
            "gamevariables" : game_var_space
        })
        

    def reset(self, **kwargs):
        # Reset the environment and get the initial observation
        observation = self.env.reset(**kwargs)
        
        if self.stack_frames:
            processed_frame = self.process_frame(observation[0]["screen"])
            self.frames.extend([processed_frame] * self.stack_size)
            
            stacked_obs = {
                "frame" : np.array(self.frames),
                "gamevariables" : observation[0]["gamevariables"]
            }
            return stacked_obs
        else:
            processed_frame = self.process_frame(observation[0]["screen"])

            return {
                "frame" : processed_frame,
                "gamevariables" : observation[0]["gamevariables"]
            }

    def step(self, action):
        # Perform the action in the environment
        observation, reward, terminated, truncated, info = self.env.step(action)
        
        processed_frame = self.process_frame(observation[0]["screen"])
        self.frames.append(processed_frame)
        
        stacked_obs = {
            "frame" : np.array(self.frames),
            "gamevariables" : observation[0]["gamevariables"]
        }

        shaped_reward = self.shape_reward(reward, observation[0]["gamevariables"])
        
        return stacked_obs, shaped_reward, terminated, truncated, info
    
    def process_frame(self, frame):
        normalized_frame = frame / 255.0
        return normalized_frame
    
    def shape_reward(self, reward, game_variables):
        return reward

    def render(self, mode="human"):
        # Render the environment as usual
        return self.env.render()
