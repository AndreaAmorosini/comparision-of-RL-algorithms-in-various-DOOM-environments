import gymnasium as gym
from stable_baselines3 import PPO, DQN, A2C
import cv2
import matplotlib.pyplot as plt
import time
from vizdoom import gymnasium_wrapper
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
import time
from customWrapper import CustomVizDoomWrapper
from gymnasium.wrappers import FlattenObservation

FRAME_SKIP = 1

# Define the path to your saved model
# MODEL_PATH = "final_models/basic/model_0.zip"
# MODEL_PATH = "final_models/center/model_42/model.zip"
MODEL_PATH = "final_models/corridor/model_43/model.zip"
# MODEL_PATH = "final_models/gathering/model_13/model.zip"


# Load the trained model
# model = PPO.load(MODEL_PATH)
# model = DQN.load(MODEL_PATH)
model = A2C.load(MODEL_PATH)

# doom_env = "VizdoomBasic-v0"
# doom_env = "VizdoomCorridor-v0"
doom_env = "VizdoomCorridor-custom-v0"
# doom_env = "VizdoomDefendCenter-v0"
# doom_env = "VizdoomDefendCenter-custom-v0"
# doom_env = "VizdoomHealthGathering-custom-v0"


# Initialize the environment for playing (same config as training)
env = gym.make(doom_env, render_mode="rgb_array", frame_skip=4)
#Only for basic env
env = CustomVizDoomWrapper(env=env, normalize=False, stack_frames=False, stack_size=1)
env = gym.wrappers.TransformReward(env, lambda r: r / 1000)
# env = gym.wrappers.RecordVideo(env, "videos/corridor/", episode_trigger=lambda x: x % 50 == 0, name_prefix="model_29")
env = gym.wrappers.HumanRendering(env)
env = gym.wrappers.ResizeObservation(env, shape=(160, 120))
env = gym.wrappers.GrayScaleObservation(env, keep_dim=True)

mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=10, render=True)

print(mean_reward)