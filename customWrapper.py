import gymnasium as gym
from gymnasium import Wrapper
from vizdoom import gymnasium_wrapper
from collections import deque
import numpy as np

class CustomVizDoomWrapper(Wrapper):
    def __init__(self, env, normalize=False, stack_frames = False, stack_size = 1):
        super().__init__(env)
        
        self.stack_frames = stack_frames
        self.normalize = normalize
        self.stack_size = stack_size
        self.frames = deque(maxlen=stack_size)
        self.initial_health = 100
        self.initial_ammo = 26
        
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
                
        self.initial_ammo = observation[0]["gamevariables"][0]
        self.initial_health = observation[0]["gamevariables"][1]
        # if isinstance(observation, list):
        #     observation = observation[0]
        
        if self.stack_frames:
            processed_frame = self.process_frame(observation[0]["screen"])
            self.frames.extend([processed_frame] * self.stack_size)
            
            stacked_obs = {
                "frame" : np.array(self.frames),
                "gamevariables" : observation[0]["gamevariables"]
            }
            return stacked_obs, {}
        else:
            processed_frame = self.process_frame(observation[0]["screen"])

            return {
                "frame" : processed_frame,
                "gamevariables" : observation[0]["gamevariables"]
            }, {}

    def step(self, action):
        # Perform the action in the environment
        observation, reward, terminated, truncated, info = self.env.step(action)
        
        processed_frame = self.process_frame(observation["screen"])
        self.frames.append(processed_frame)
        
        stacked_obs = {
            "frame" : np.array(self.frames),
            "gamevariables" : observation["gamevariables"]
        }

        shaped_reward = self.shape_reward(reward, observation["gamevariables"])
        
        return stacked_obs, shaped_reward, terminated, truncated, info
    
    def process_frame(self, frame):
        if self.normalize:
            frame = frame/255.0
            return frame
        else:
            return frame
    
    def post_process_frame(self, frame):
        return (frame * 255).astype(np.uint8)
    
    def shape_reward(self, reward, game_variables):
        ammo = game_variables[0]
        health = game_variables[1]
        
        # if ammo < self.initial_ammo:
        #     reward -= 0.1
            
        if health < self.initial_health:
            reward -= 0.01
          
        return reward

    def render(self, mode="human"):
        # Render the environment as usual
        return self.env.render()
