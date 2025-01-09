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
from argparse import ArgumentParser
import gymnasium
import os

AVAILABLE_ENVS = [env for env in gymnasium.envs.registry.keys() if "Vizdoom" in env]

def main(args):
    env_paths = {
        "HealthGathering": "gathering",
        "DefendTheCenter": "center",
        "DeadlyCorridor": "corridor",
    }
    env_path = env_paths[args.env]
    MODEL_PATH = f"final_models/{env_path}/model_{args.model_number_id}/model.zip"
    
    if not os.path.exists(MODEL_PATH):
        print(f"Model {MODEL_PATH} does not exist.")
    
    if args.model == "PPO":
        model = PPO.load(MODEL_PATH)
    elif args.model == "DQN":
        model = DQN.load(MODEL_PATH)
    elif args.model == "A2C":
        model = A2C.load(MODEL_PATH)

    envs = {
        "HealthGathering": "VizdoomHealthGathering-custom-v0",
        "DefendTheCenter": "VizdoomDefendCenter-custom-v0",
        "DeadlyCorridor": "VizdoomCorridor-custom-v0",
    }
    ENV = envs[args.env]

    # Initialize the environment for playing (same config as training)
    env = gym.make(ENV, render_mode="rgb_array", frame_skip=4)
    env = CustomVizDoomWrapper(env=env, normalize=False)
    if args.env == "DeadlyCorridor":
        env = gym.wrappers.TransformReward(env, lambda r: r / 1000)
    # env = gym.wrappers.TransformReward(env, lambda r: r / 1000)
    env = gym.wrappers.HumanRendering(env)
    env = gym.wrappers.ResizeObservation(env, shape=(160, 120))
    env = gym.wrappers.GrayScaleObservation(env, keep_dim=True)

    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=10, render=True)

    print(mean_reward)


if __name__ == "__main__":
    parser = ArgumentParser("Evaluate and Display stable-baselines3 agents on ViZDoom environments.")
    parser.add_argument(
        "--env",
        default="DeadlyCorridor",
        choices=["HealthGathering", "DefendTheCenter", "DeadlyCorridor"],
        help="Name of the environment to play",
    )
    parser.add_argument(
        "--model",
        default="PPO",
        choices=["PPO", "DQN", "A2C"],
        help="Name of the model to train",
    )
    parser.add_argument(
        "--model_number_id",
        default=38,
        type=int,
        help="Model Number ID",
    )
    parser.add_argument(
        "--eval_episodes",
        default=10,
        type=int,
        help="Number of episodes to evaluate the model",
    )
    args = parser.parse_args()

    try:
        main(args)
    except KeyboardInterrupt:
        print("Evaluation interrupted")
