from argparse import ArgumentParser

import cv2
import gymnasium
import gymnasium.wrappers.frame_stack
import gymnasium.wrappers.human_rendering
import gymnasium.wrappers.resize_observation
import numpy as np
import stable_baselines3
from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common import evaluation, policies
import os
import wandb
from wandb.integration.sb3 import WandbCallback
from customWrapper import CustomVizDoomWrapper
import vizdoom.gymnasium_wrapper  # noqa

AVAILABLE_ENVS = [env for env in gymnasium.envs.registry.keys() if "Vizdoom" in env]
TRAINING_TIMESTEPS = 1000000

best_params = {
    "DeadlyCorridor": {
        "PPO": {
            "n_steps": 1317,
            "learning_rate": 0.00014084502905743732,
            "gamma": 0.9347573535833384,
            "n_epochs": 3,
            "batch_size": 3,
            "ent_coef": 1.2421501758561545e-08,
        },
        "DQN": {
            "learning_rate": 0.00012959243522583412,
            "gamma": 0.7854337165235085,
            "batch_size": 68,
            "train_freq": 3,
            "gradient_steps": 9,
        },
        "A2C": {
            "learning_rate": 0.0001539966506155027,
            "gamma": 0.95,
            "n_steps": 32,
            "gae_lambda": 0.9,
            "max_grad_norm": 5.0,
        },
    },
    "DefendTheCenter": {
        "PPO": {
            "n_steps": 1685,
            "learning_rate": 0.00011967127960181731,
            "gamma": 0.9124892849701195,
            "n_epochs": 1,
            "batch_size": 7,
            "ent_coef": 0.0015922188908024028,
        },
        "DQN": {
            "learning_rate": 0.00020084665886895524,
            "gamma": 0.338511413266372,
            "batch_size": 84,
            "train_freq": 1,
            "gradient_steps": 4,
        },
        "A2C": {
            "learning_rate": 0.09220377006945718,
            "gamma": 0.999,
            "n_steps": 8,
            "gae_lambda": 0.8,
            "max_grad_norm": 5.0,
        },
    },
    "HealthGathering": {
        "PPO": {
            "n_steps": 1674,
            "learning_rate": 1.0443025107862082e-05,
            "gamma": 0.9278925638444294,
            "n_epochs": 3,
            "batch_size": 25,
            "ent_coef": 0.0009994107100674417,
        },
        "DQN": {
            "learning_rate": 5.624336017055847e-05,
            "gamma": 0.6222850640947339,
            "batch_size": 61,
            "train_freq": 4,
            "gradient_steps": 4,
        },
        "A2C": {
            "learning_rate": 0.002314814844294784,
            "gamma": 0.99,
            "n_steps": 128,
            "gae_lambda": 1.0,
            "max_grad_norm": 0.8,
        },
    },
}


def main(args):
    env_paths = {
        "HealthGathering": "gathering",
        "DefendTheCenter": "center",
        "DeadlyCorridor": "corridor",
    }
    env_path = env_paths[args.env]

    
    LOG_DIR = f"./logs/{env_path}"
    MODEL_SAVE_DIR = f"./final_models/{env_path}"
    
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
        
    if not os.path.exists(MODEL_SAVE_DIR):
        os.makedirs(MODEL_SAVE_DIR)

    
    MODEL = args.model
    environment = args.env
    USE_BEST_PARAMS = args.use_best_params
    
    envs = {
        "HealthGathering": "VizdoomHealthGathering-custom-v0",
        "DefendTheCenter": "VizdoomDefendCenter-custom-v0",
        "DeadlyCorridor": "VizdoomCorridor-custom-v0",
    }
    ENV = envs[environment]

    # For PPO
    if MODEL == "PPO":
        config = {
            "MODEL": MODEL,
            "env": environment,
            "training_timesteps": TRAINING_TIMESTEPS,
            "n_steps": 2048 if not USE_BEST_PARAMS else best_params[environment][MODEL]["n_steps"],
            "learning_rate": 1e-4 if not USE_BEST_PARAMS else best_params[environment][MODEL]["learning_rate"],
            "gamma": 0.99 if not USE_BEST_PARAMS else best_params[environment][MODEL]["gamma"],
            "n_epochs": 10 if not USE_BEST_PARAMS else best_params[environment][MODEL]["n_epochs"],
            "batch_size": 64 if not USE_BEST_PARAMS else best_params[environment][MODEL]["batch_size"],
            "ent_coef": 0.0 if not USE_BEST_PARAMS else best_params[environment][MODEL]["ent_coef"],
        }
    elif MODEL == "DQN":
        # For DQN
        config = {
            "MODEL": MODEL,
            "env": environment,
            "training_timesteps": TRAINING_TIMESTEPS,
            "learning_rate": 1e-4 if not USE_BEST_PARAMS else best_params[environment][MODEL]["learning_rate"],
            "gamma": 0.99 if not USE_BEST_PARAMS else best_params[environment][MODEL]["gamma"],
            "batch_size": 32 if not USE_BEST_PARAMS else best_params[environment][MODEL]["batch_size"],
            "buffer_size": int(TRAINING_TIMESTEPS / 40),
            "train_freq": 4 if not USE_BEST_PARAMS else best_params[environment][MODEL]["train_freq"],
            "learning_starts": int(TRAINING_TIMESTEPS / 20),
            "gradient_steps": 1 if not USE_BEST_PARAMS else best_params[environment][MODEL]["gradient_steps"],
        }
    elif MODEL == "A2C":
        # FOR A2C
        config = {
            "MODEL": MODEL,
            "env": environment,
            "training_timesteps": TRAINING_TIMESTEPS,
            "learning_rate": 7e-4 if not USE_BEST_PARAMS else best_params[environment][MODEL]["learning_rate"],
            "gamma": 0.99 if not USE_BEST_PARAMS else best_params[environment][MODEL]["gamma"],
            "n_steps": 5 if not USE_BEST_PARAMS else best_params[environment][MODEL]["n_steps"],
            "gae_lambda": 1.0 if not USE_BEST_PARAMS else best_params[environment][MODEL]["gae_lambda"],
            "max_grad_norm": 0.5 if not USE_BEST_PARAMS else best_params[environment][MODEL]["max_grad_norm"],
        }

    def wrap_env(env):
        env = CustomVizDoomWrapper(
            env, normalize=False
        )
        if args.env == "DeadlyCorridor":
            env = gymnasium.wrappers.TransformReward(env, lambda r: r / 1000)
        env = gymnasium.wrappers.HumanRendering(env)
        env = gymnasium.wrappers.ResizeObservation(env, shape=(160, 120))
        env = gymnasium.wrappers.GrayScaleObservation(env, keep_dim=True)
        return env

    models = os.listdir(MODEL_SAVE_DIR)

    if models == []:
        nrModel = 0
    else:
        nrModel = len(models) + 1

    envs = make_vec_env(
        ENV,
        n_envs=2,
        wrapper_class=wrap_env,
        env_kwargs={"frame_skip": 4},
    )

    config["RESOLUTION"] = (
        str(envs.observation_space.shape[0])
        + "x"
        + str(envs.observation_space.shape[1])
    )
    config["COLOR_SPACE"] = "GRAY" if envs.observation_space.shape[2] == 1 else "RGB"

    # Initialize wandb
    if args.use_wandb:
        run = wandb.init(
            project="vizdoom",
            name=f"vizdoom_{env_path}_" + str(nrModel),
            group=env_path,
            tags=[
                MODEL,
                "vizdoom",
                environment,
                config["RESOLUTION"],
                config["COLOR_SPACE"],
                "BestParams" if USE_BEST_PARAMS else "Baseline",
            ],
            config=config,
            sync_tensorboard=True,
            save_code=True,
            monitor_gym=True,
        )

    if MODEL == "PPO":
        agent = PPO(
            policies.ActorCriticCnnPolicy,
            envs,
            n_steps=config["n_steps"],
            verbose=1,
            tensorboard_log=LOG_DIR + "/" + run.id,
            learning_rate=config["learning_rate"],
            n_epochs=config["n_epochs"],
            batch_size=config["batch_size"],
            gamma=config["gamma"],
        )
    elif MODEL == "DQN":
        agent = DQN(
            stable_baselines3.dqn.CnnPolicy,
            envs,
            learning_rate=config["learning_rate"],
            batch_size=config["batch_size"],
            gamma=config["gamma"],
            tensorboard_log=LOG_DIR + "/" + run.id,
            verbose=1,
            buffer_size=config["buffer_size"],
            train_freq=config["train_freq"],
        )
    elif MODEL == "A2C":
        agent = A2C(
            policies.ActorCriticCnnPolicy,
            envs,
            learning_rate=config["learning_rate"],
            n_steps=config["n_steps"],
            gamma=config["gamma"],
            gae_lambda=config["gae_lambda"],
            verbose=1,
            tensorboard_log=LOG_DIR + "/" + run.id,
        )

    if args.use_wandb:
        agent.learn(
            total_timesteps=TRAINING_TIMESTEPS,
            callback=WandbCallback(
                model_save_path=f"{MODEL_SAVE_DIR}/model_{nrModel}",
                verbose=1,
            ),
        )
    else:
        agent.learn(total_timesteps=TRAINING_TIMESTEPS)

    if args.use_wandb:
        run.finish()


if __name__ == "__main__":
    parser = ArgumentParser("Train stable-baselines3 agents on ViZDoom environments.")
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
        "--use_best_params",
        default=False,
        action="store_true",
        help="Use best hyperparameters for the model or the default ones",
    )
    parser.add_argument(
        "--use_wandb",
        default=False,
        action="store_true",
        help="Use Weights and Biases for logging",
    )
    args = parser.parse_args()

    try:
        main(args)
    except KeyboardInterrupt:
        print("Training interrupted")
        # run.finish()
