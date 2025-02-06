import gymnasium as gym
from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.monitor import Monitor
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
    
    if (args.use_baseline_model and args.use_best_model) or (args.use_baseline_model and args.model_number_id) or (args.use_best_model and args.model_number_id) or (args.use_baseline_model and args.use_best_model and args.model_number_id):
        print("Si e' pregati di selezionare solo un modello da visualizzare.")
    
    if args.use_baseline_model:
        MODEL_PATH = f"final_models1/{env_path}/{args.model}_Baseline/model.zip"
        VIDEO_PATH = f"videos/{args.env}/{args.model}/{args.model}_Baseline/"
    elif args.use_best_model:
        MODEL_PATH = f"final_models1/{env_path}/{args.model}_BestParams/model.zip"
        VIDEO_PATH = f"videos/{args.env}/{args.model}/{args.model}_BestParams/"
    else:
        MODEL_PATH = f"final_models/{env_path}/model_{args.model_number_id}/model.zip"
        VIDEO_PATH = f"videos/{args.env}/{args.model}/model_{args.model_number_id}/"
    
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
    if args.record_video:
        env = gym.wrappers.RecordVideo(env, video_folder=VIDEO_PATH)
    else:
        env = gym.wrappers.HumanRendering(env)
    env = gym.wrappers.ResizeObservation(env, shape=(160, 120))
    env = gym.wrappers.GrayScaleObservation(env, keep_dim=True)
    env = Monitor(env)

    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=args.eval_episodes, render=True)

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
        # default=38,
        type=int,
        help="Model Number ID",
    )
    parser.add_argument(
        "--eval_episodes",
        default=10,
        type=int,
        help="Number of episodes to evaluate the model",
    )
    parser.add_argument(
        "--use_baseline_model",
        default=False,
        action="store_true",
        help="Use the baseline model for evaluation",
    )
    parser.add_argument(
        "--use_best_model",
        default=False,
        action="store_true",
        help="Use the best model for evaluation",
    )
    parser.add_argument(
        "--record_video",
        default=False,
        action="store_true",
        help="Record video of the evaluation",
    )
    args = parser.parse_args()

    try:
        main(args)
    except KeyboardInterrupt:
        print("Evaluation interrupted")
