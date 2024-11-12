import gymnasium as gym
from stable_baselines3 import PPO
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


IMAGE_SHAPE = (60, 80)
FRAME_SKIP = 1


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

    def __init__(self, env, shape=IMAGE_SHAPE):
        super().__init__(env)
        self.image_shape = shape
        self.image_shape_reverse = shape[::-1]
        self.env.frame_skip = FRAME_SKIP
        
        assert isinstance(self.image_shape_reverse, tuple) and all(
            isinstance(dim, int) and dim > 0 for dim in self.image_shape_reverse
        ), f"Invalid target shape: {self.image_shape_reverse}"

        # Create new observation space with the new shape
        print("Observation Space")
        print(env.observation_space)
        num_channels = env.observation_space["frame"].shape[-1]
        new_shape = (shape[0], shape[1], num_channels)
        self.observation_space = gym.spaces.Box(
            0, 255, shape=new_shape, dtype=np.uint8
        )
        
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



# Define the path to your saved model
# MODEL_PATH = "final_models/basic/model_0.zip"
# MODEL_PATH = "final_models/center/model_22/model.zip"
MODEL_PATH = "final_models/corridor/model_12/model.zip"


# Load the trained model
model = PPO.load(MODEL_PATH)

# doom_env = "VizdoomBasic-v0"
# doom_env = "VizdoomCorridor-v0"
doom_env = "VizdoomCorridor-custom-v0"
# doom_env = "VizdoomDefendCenter-v0"
# doom_env = "VizdoomDefendCenter-custom-v0"


# Initialize the environment for playing (same config as training)
env = gym.make(doom_env, render_mode="rgb_array", frame_skip=4)
#Only for basic env
env = CustomVizDoomWrapper(env=env, normalize=True, stack_frames=False, stack_size=1)
# env = ObservationWrapper(env)
# env = gym.wrappers.FrameStackObservation(env, num_stack=4, padding_type = "zero")
# env = gym.wrappers.NormalizeObservation(env)
env = FlattenObservation(env)
env = gym.wrappers.TransformReward(env, lambda r: r * 0.01)
env = gym.wrappers.HumanRendering(env)

mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=100, render=True)

print(mean_reward)




# # Loop to test agent without using evaluate_policy
# for episode in range(100):
#     obs, info = env.reset()  # observations
#     done = False  # havent completed
#     total_reward = 0
#     while not done:
#         action, _ = model.predict(obs)  # get action
#         obs, reward, done, truncated, info = env.step(action)  # pass into env step function
#         # time.sleep(0.20)
#         total_reward += reward  # update reward
#     print("Total Reward for episode {} is {}".format(episode, total_reward))

