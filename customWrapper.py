import gymnasium as gym
from gymnasium import Wrapper
from vizdoom import gymnasium_wrapper
from collections import deque
import numpy as np
import cv2
from gymnasium.vector.utils import batch_space, concatenate, create_empty_array
from gymnasium.core import ObsType
from copy import deepcopy

class CustomVizDoomWrapper(Wrapper):
    def __init__(self, env, normalize=False, stack_frames = False, stack_size = 1, padding_type: ObsType = "reset", env_name = None):
        super().__init__(env)
        
        print("AVAILABLE GAME VARIABLES")
        print(env.game.get_available_game_variables())
        
        print(env.spec.id)
        
        self.env_id = env.spec.id
        self.stack_frames = stack_frames
        self.normalize = normalize
        self.stack_size = stack_size
        self.frames = deque(maxlen=stack_size)
        self.initial_health = 100
        self.initial_ammo = 26
        self.killcount = 0
        self.health = 100
        self.ammo = 26
        
        image_space = env.observation_space["screen"]
        game_var_space = env.observation_space["gamevariables"]
        
        image_shape = image_space.shape        
        
        if self.stack_frames:
            stacked_image_shape = (stack_size, *image_shape)
        else:
            stacked_image_shape = image_shape
        # self.observation_space = gym.spaces.Dict({
        #     "frame" : gym.spaces.Box(low=0, high=255, shape=(stacked_image_shape), dtype=np.uint8),
        #     "gamevariables" : game_var_space
        # })
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(stacked_image_shape), dtype=np.uint8)

    def reset(self, **kwargs):
        # Reset the environment and get the initial observation
        observation = self.env.reset(**kwargs)
                
        # self.initial_ammo = observation[0]["gamevariables"][0]
        self.initial_health = 100
        self.health = self.initial_health
        self.killcount = 0
        self.initial_ammo = observation[0]["gamevariables"][2]
        self.ammo = self.initial_ammo
        # if isinstance(observation, list):
        #     observation = observation[0]
        
        if self.stack_frames:
            processed_frame = self.process_frame(observation[0]["screen"])
            self.frames.append(processed_frame)
            
            
            stacked_obs = {
                "frame" : np.array(self.frames),
                "gamevariables" : observation[0]["gamevariables"]
            }
            
            
            return stacked_obs, {}
        else:
            processed_frame = self.process_frame(observation[0]["screen"])

            # return {
            #     "frame" : processed_frame,
            #     "gamevariables" : observation[0]["gamevariables"]
            # }, {}
            return processed_frame, {}

    def step(self, action):
        # Perform the action in the environment
        observation, reward, terminated, truncated, info = self.env.step(action)
        
        processed_frame = self.process_frame(observation["screen"])
        
        
        if self.stack_frames:
            self.frames.append(processed_frame)
            stacked_obs = {
                "frame" : np.array(self.frames),
                "gamevariables" : observation["gamevariables"]
            }      
        else:
            stacked_obs = {
                "frame" : processed_frame,
                "gamevariables" : observation["gamevariables"]
            }      

        shaped_reward = self.shape_reward(reward, observation["gamevariables"])
        
        return processed_frame, shaped_reward, terminated, truncated, info
    
    def process_frame(self, frame):
        
        # if len(frame.shape) == 2:
        #     frame = frame[..., np.newaxis]
        
        if self.normalize:
            frame = frame/255.0
            
        return frame
    
    def post_process_frame(self, frame):
        return (frame * 255).astype(np.uint8)
    
    def shape_reward(self, reward, game_variables):
        # ammo = game_variables[0]
        health = game_variables[0]
        killcount = game_variables[1]
        ammo = game_variables[2]
                
        current_reward = reward
        
        # if ammo < self.initial_ammo:
        #     reward -= 0.1
            
        # if health < self.initial_health:
        #     reward -= 0.01
        
        #Reward for staying with as much health as possible
        # health_reward = health * 1
        # reward += health_reward
        
        # Penalty for losing health
        # healthDiff = self.health - health
        # health_penalty = healthDiff * 50
        # self.health = health
        # reward -= health_penalty
        
        #Rewards for Deadly Corridor
        if "VizdoomCorridor" in self.env_id:
        
            health_delta = health - self.health
            self.health = health
            
            ammo_delta = ammo - self.ammo
            self.ammo = ammo
            
            killcount_delta = killcount - self.killcount
            self.killcount = killcount
            
            if health_delta < 0:
                current_reward -= 5
            
            if ammo_delta != 0:
                current_reward += (ammo_delta * 0.5)
                
            if killcount_delta > 0:
                current_reward += (killcount_delta * 100)
                
        if "VizdoomDefendCenter" in self.env_id:
            
            health_delta = health - self.health
            self.health = health
            
            ammo_delta = ammo - self.ammo
            self.ammo = ammo
            
            if health_delta < 0:
                current_reward -= 0.5
            
            if ammo_delta != 0:
                current_reward += (ammo_delta * 0.25)


        
        #Reward for increasing killcount
        # if killcount > self.killcount:
        #     reward += 1
        #     self.killcount = killcount

        #Reward for killcount incremental
        # self.killcount = killcount
        # killcount_reward = killcount * 1.5
        # reward += killcount_reward
        
        #Reward for incrementing killcount
        # if killcount > self.killcount:
        #     reward += ((self.killcount - killcount) * 50)
        #     self.killcount = killcount
            
        # time_penalty = 0.01
        # reward -= time_penalty
        
        # survival_reward = 0.1
        # reward += survival_reward
            
        #Kill multiplier    
        # self.killcount = killcount
        # if killcount > 0 and reward >= 0:
        #     reward *= killcount
        # elif killcount == 0:
        #     reward += 100

          
        return current_reward
    
    def grayscale(self, frame):
        gray = cv2.cvtColor(np.moveaxis(frame, 0, 1), cv2.COLOR_RGB2GRAY)
        return frame

    def render(self, mode="human"):
        # Render the environment as usual
        return self.env.render()


IMAGE_SHAPE = (60, 80)
FRAME_SKIP = 4

class ObservationWrapper(gym.ObservationWrapper):
    """
    ViZDoom environments return dictionaries as observations, containing
    the main image as well other info.
    The image is also too large for normal training.

    This wrapper replaces the dictionary observation space with a simple
    Box space (i.e., only the RGB image), and also resizes the image to a
    smaller size.

    NOTE: Ideally, you should set the image size to smaller in the scenario files
          for faster running of ViZDoom. This can really impact performance,
          and this code is pretty slow because of this!
    """

    def __init__(self, env, shape=IMAGE_SHAPE, frame_skip=FRAME_SKIP):
        super().__init__(env)
        self.image_shape = shape
        self.image_shape_reverse = shape[::-1]
        self.env.frame_skip = frame_skip

        assert isinstance(self.image_shape_reverse, tuple) and all(
            isinstance(dim, int) and dim > 0 for dim in self.image_shape_reverse
        ), f"Invalid target shape: {self.image_shape_reverse}"

        # Create new observation space with the new shape
        print("Observation Space")
        print(env.observation_space)
        num_channels = env.observation_space["frame"].shape[-1]
        new_shape = (shape[0], shape[1], num_channels)
        self.observation_space = gym.spaces.Box(0, 255, shape=new_shape, dtype=np.uint8)

    def post_process_frame(self, frame):
        return (frame * 255).astype(np.uint8)

    def process_frame(self, frame):
        normalized_frame = frame / 255.0
        return normalized_frame

    def observation(self, observation):
        print("Observation")
        print(observation)
        # observation = self.post_process_frame(observation["frame"])
        observation = observation["frame"]
        observation = cv2.resize(observation, self.image_shape_reverse)
        return observation
