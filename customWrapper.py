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
    def __init__(self, env, normalize=False, padding_type: ObsType = "reset", env_name = None):
        super().__init__(env)
        
        print("AVAILABLE GAME VARIABLES")
        print(env.game.get_available_game_variables())
        
        print(env.spec.id)
        
        self.env_id = env.spec.id
        self.normalize = normalize
        self.initial_health = 100
        self.initial_ammo = 26
        self.killcount = 0
        self.health = 100
        self.ammo = 26
        self.armor = 0
        
        image_space = env.observation_space["screen"]
        game_var_space = env.observation_space["gamevariables"]
        
        image_shape = image_space.shape        
        
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(image_shape), dtype=np.uint8)

    def reset(self, **kwargs):
        # Reset the environment and get the initial observation
        observation = self.env.reset(**kwargs)
                
        self.initial_health = 100
        self.health = self.initial_health
        self.killcount = 0
        self.initial_ammo = observation[0]["gamevariables"][2]
        self.ammo = self.initial_ammo
        if "VizdoomDeathmatch" in self.env_id:
            self.armor = observation[0]["gamevariables"][3]
                
        processed_frame = self.process_frame(observation[0]["screen"])
        return processed_frame, {}

    def step(self, action):
        # Perform the action in the environment
        observation, reward, terminated, truncated, info = self.env.step(action)
        
        processed_frame = self.process_frame(observation["screen"])             
            
        # if terminated:
        #     print("KILLCOUNT " + str(self.killcount))     

        shaped_reward = self.shape_reward(reward, observation["gamevariables"])
        
        return processed_frame, shaped_reward, terminated, truncated, info
    
    def process_frame(self, frame):        
        if self.normalize:
            frame = frame/255.0
            
        return frame
    
    def post_process_frame(self, frame):
        return (frame * 255).astype(np.uint8)
    
    def shape_reward(self, reward, game_variables):
        health = game_variables[0]
        killcount = game_variables[1]
        ammo = game_variables[2]
                
        current_reward = reward
            
        #Rewards for Deadly Corridor
        if "VizdoomCorridor" in self.env_id:
            
            current_reward = current_reward / 5
        
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
            
            self.killcount = killcount
            
            if health_delta < 0:
                current_reward -= 0.5
            
            if ammo_delta != 0:
                current_reward += (ammo_delta * 0.25)
             
             #Bonus for not wasting ammo   
            # optimal_ammo = self.initial_ammo - killcount
            # delta_optimal_ammo = optimal_ammo - ammo
            # if delta_optimal_ammo < 3:
            #     current_reward += 10

        if "VizdoomHealthGathering" in self.env_id:
            health_delta = health - self.health
            self.health = health
            
            # print("HEALTH DELTA " + str(health_delta))
            
            # if health_delta > 0:
            #     current_reward += health_delta
                
            if health_delta > 0:
                current_reward += 5
                
            # if health_delta < 0:
            #     current_reward -= 0.1
                                    
            # if health_delta < 0:
            #     current_reward -= 5
            
        if "VizdoomDeathmatch" in self.env_id:
            
            current_reward = 0
            
            armor = game_variables[3] 
            
            if killcount > self.killcount:
                current_reward += 1
                self.killcount = killcount
                
            if health < self.health:
                current_reward -= 0.1
                
            if health > self.health:
                current_reward += 0.1
                
            if armor < self.armor:
                current_reward -= 0.1
            
            if armor > self.armor:
                current_reward += 0.1
          
        return current_reward
    
    def grayscale(self, frame):
        gray = cv2.cvtColor(np.moveaxis(frame, 0, 1), cv2.COLOR_RGB2GRAY)
        return frame

    def render(self, mode="human"):
        # Render the environment as usual
        return self.env.render()
