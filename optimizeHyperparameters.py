import sys
import os
from argparse import ArgumentParser

import cv2
import gymnasium
import gymnasium.wrappers.human_rendering
from gymnasium.wrappers import FlattenObservation
import numpy as np
import stable_baselines3
from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
import os
import wandb
from wandb.integration.sb3 import WandbCallback
from customWrapper import CustomVizDoomWrapper, ObservationWrapper
from stable_baselines3.common import evaluation, policies
import optuna
import optuna_dashboard


import vizdoom.gymnasium_wrapper  # noqa


AVAILABLE_ENVS = [env for env in gymnasium.envs.registry.keys() if "Vizdoom" in env]

TRAINING_TIMESTEPS = 1000000


def main(args):
    STUDY_NAME = f"{args.model}_study"
    LOG_DIR = f"./optimization/logs/{args.env}/{args.model}"
    DB_DIR = f"./optimization/DB/{args.env}/{args.model}"
    
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
        
    if not os.path.exists(DB_DIR):
        os.makedirs(DB_DIR)
    
    envs = {
        "HealthGathering": "VizdoomHealthGathering-custom-v0",
        "DefendTheCenter": "VizdoomDefendCenter-custom-v0",
        "DeadlyCorridor": "VizdoomDeadlyCorridor-custom-v0",
    }
    ENV = envs[args.env]

    
    def optimizeParams(model, trial):
        # For PPO
        if model == "PPO":
            config = {
                "n_steps": int(trial.suggest_loguniform("n_steps", 1024, 8129)),
                "learning_rate": trial.suggest_loguniform("learning_rate", 1e-5, 0.01),
                "gamma": trial.suggest_loguniform("gamma", 0.9, 0.9999),
                "n_epochs": int(trial.suggest_loguniform("n_epochs", 1, 15)),
                "batch_size": int(trial.suggest_loguniform("batch_size", 2, 48)),
                "ent_coef": trial.suggest_loguniform("ent_coef", 1e-8, 1e-1),
            }
        elif model == "DQN":
            # For DQN
            config = {
                "learning_rate": trial.suggest_loguniform("learning_rate", 1e-5, 1),
                "gamma": trial.suggest_float("gamma", 0, 1.0),
                "batch_size": trial.suggest_int("batch_size", 2, 128),
                "buffer_size": int(TRAINING_TIMESTEPS / 40),
                "learning_starts": int(TRAINING_TIMESTEPS / 20),
                "train_freq": (trial.suggest_int("train_freq", 1, 10), "episode"),
                "gradient_steps": trial.suggest_int("gradient_steps", 1, 10),
            }
        elif model == "A2C":
            # FOR A2C
            config = {
                "learning_rate": trial.suggest_loguniform("learning_rate", 1e-5, 1),
                "gamma": trial.suggest_categorical(
                    "gamma", [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999]
                ),
                "n_steps": trial.suggest_categorical(
                    "n_steps", [8, 16, 32, 64, 128, 256, 512, 1024, 2048]
                ),
                "gae_lambda": trial.suggest_categorical(
                    "gae_lambda", [0.8, 0.9, 0.92, 0.95, 0.98, 0.99, 1.0]
                ),
                "max_grad_norm": trial.suggest_categorical(
                    "max_grad_norm", [0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0, 5.0]
                ),
            }

        return config


    def optimize_agent(trial):
        model_params = optimizeParams(args.model, trial)
        RUN_NAME = f"Trial_{trial.number}_"

        for key in model_params:
            RUN_NAME += f"{key}={model_params[key]}_"
            RUN_NAME = RUN_NAME[:-1]

        def wrap_env(env):
            env = CustomVizDoomWrapper(
                env, normalize=False
            )
            if args.env == "DeadlyCorridor":
                env = gymnasium.wrappers.TransformReward(env, lambda r: r / 1000)
            env = gymnasium.wrappers.ResizeObservation(env, shape=(160, 120))
            env = gymnasium.wrappers.GrayScaleObservation(env, keep_dim=True)
            return env

        envs = make_vec_env(
            ENV,
            n_envs=1,
            wrapper_class=wrap_env,
            env_kwargs={"frame_skip": 4},
        )

        MODEL = args.model
        if MODEL == "PPO":
            agent = PPO(
                policies.ActorCriticCnnPolicy,
                envs,
                verbose=0,
                tensorboard_log=LOG_DIR,
                **model_params,
            )
        elif MODEL == "DQN":
            agent = DQN(
                stable_baselines3.dqn.CnnPolicy,
                envs,
                tensorboard_log=LOG_DIR,
                verbose=0,
                **model_params,
            )
        elif MODEL == "A2C":
            agent = A2C(
                policies.ActorCriticCnnPolicy,
                envs,
                verbose=0,
                tensorboard_log=LOG_DIR,
                **model_params,
            )

        logger = configure(LOG_DIR + "/" + RUN_NAME, ["stdout", "csv", "tensorboard"])
        agent.set_logger(logger)

        agent.learn(
            total_timesteps=TRAINING_TIMESTEPS,
            log_interval=20,
        )

        mean_reward, _ = evaluation.evaluate_policy(agent, envs, n_eval_episodes=10)
        return mean_reward

    study = optuna.create_study(
        direction="maximize",
        study_name=STUDY_NAME,
        storage="sqlite:///" + DB_DIR + ".db",
        load_if_exists=True,
    )
    study.optimize(optimize_agent, n_trials=args.n_trials, gc_after_trial=True)


if __name__ == "__main__":
    parser = ArgumentParser("Optimize Hyperparamethers of stable-baselines3 agents on ViZDoom environments.")
    parser.add_argument(
        "--env",
        default="DeadlyCorridor",
        choices=AVAILABLE_ENVS,
        help="Name of the environment to play [HealthGathering, DefendTheCenter, DeadlyCorridor]",
    )
    parser.add_argument(
        "--model",
        default="PPO",
        choices=["PPO", "DQN", "A2C"],
        help="Name of the model to train [PPO, DQN, A2C]",
    )
    parser.add_argument(
        "--n_trials",
        default=50,
        type=int,
        help="Number of trials to try",
    )
    args = parser.parse_args()
    
    try:
        main(args)
    except KeyboardInterrupt:
        print("Optimization Interrupted")
