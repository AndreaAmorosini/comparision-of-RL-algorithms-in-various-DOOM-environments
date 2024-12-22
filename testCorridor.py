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
import os
import wandb
from wandb.integration.sb3 import WandbCallback
from customWrapper import CustomVizDoomWrapper, ObservationWrapper
from stable_baselines3.common import evaluation, policies


import vizdoom.gymnasium_wrapper  # noqa


# DEFAULT_ENV = "VizdoomCorridor-v0"
DEFAULT_ENV = "VizdoomCorridor-custom-v0"
AVAILABLE_ENVS = [env for env in gymnasium.envs.registry.keys() if "Vizdoom" in env]

# MODEL = "PPO"
# MODEL = "DQN"
MODEL = "A2C"
TRAINING_TIMESTEPS = 1000000
USE_BEST_PARAMS = True

N_ENVS = 1
FRAME_SKIP = 4


if MODEL == "PPO":
    # Training parameters for PPO
    N_STEPS = 2048 if not USE_BEST_PARAMS else 1317
    LEARNING_RATE = 1e-4 if not USE_BEST_PARAMS else 0.00014084502905743732
    N_EPOCHS = 10 if not USE_BEST_PARAMS else 3
    BATCH_SIZE = 64 if not USE_BEST_PARAMS else 3
    GAMMA = 0.99 if not USE_BEST_PARAMS else 0.9347573535833384
    ENT_COEF = 0.0 if not USE_BEST_PARAMS else 1.2421501758561545e-08
elif MODEL == "DQN":
    # Training parameters for DQN
    LEARNING_RATE = 1e-4 if not USE_BEST_PARAMS else 0.00012959243522583412
    BATCH_SIZE = 32 if not USE_BEST_PARAMS else 68
    GAMMA = 0.99 if not USE_BEST_PARAMS else 0.7854337165235085
    BUFFER_SIZE = int(TRAINING_TIMESTEPS / 40)
    TRAIN_FREQ = 4 if not USE_BEST_PARAMS else 3
    LEARNING_STARTS = int(TRAINING_TIMESTEPS / 20)
    GRADIENT_STEPS = 1 if not USE_BEST_PARAMS else 9
elif MODEL == "A2C":
    # Training parameters for A2C
    LEARNING_RATE = 1e-4 if not USE_BEST_PARAMS else 0.0001539966506155027
    GAMMA = 0.99 if not USE_BEST_PARAMS else 0.95
    N_STEPS = 5 if not USE_BEST_PARAMS else 32
    GAE_LAMBDA = 1.0 if not USE_BEST_PARAMS else 0.9
    MAX_GRAD_NORM = 0.5 if not USE_BEST_PARAMS else 5.0


CHECKPOINT_DIR = "./checkpoints/train/corridor"
LOG_DIR = "./logs/corridor"
MODEL_SAVE_DIR = "./final_models/corridor"

# For PPO
if MODEL == "PPO":
    config = {
        "MODEL": MODEL,
        "env": DEFAULT_ENV,
        "training_timesteps": TRAINING_TIMESTEPS,
        "n_steps": N_STEPS,
        "n_envs": N_ENVS,
        "frame_skip": FRAME_SKIP,
        "learning_rate": LEARNING_RATE,
        "gamma": GAMMA,
        "n_epochs": N_EPOCHS,
        "batch_size": BATCH_SIZE,
        "ent_coef": ENT_COEF,
        "RESOLUTION": "160x120",
        "COLOR_SPACE": "GRAY",
    }
elif MODEL == "DQN":
    # For DQN
    config = {
        "MODEL": MODEL,
        "env": DEFAULT_ENV,
        "training_timesteps": TRAINING_TIMESTEPS,
        "frame_skip": FRAME_SKIP,
        "learning_rate": LEARNING_RATE,
        "gamma": GAMMA,
        "batch_size": BATCH_SIZE,
        "RESOLUTION": "160x120",
        "COLOR_SPACE": "GRAY",
        "buffer_size": BUFFER_SIZE,
        "train_freq": TRAIN_FREQ,
    }
elif MODEL == "A2C":
    # FOR A2C
    config = {
        "MODEL": MODEL,
        "env": DEFAULT_ENV,
        "training_timesteps": TRAINING_TIMESTEPS,
        "frame_skip": FRAME_SKIP,
        "learning_rate": LEARNING_RATE,
        "gamma": GAMMA,
        "n_steps": N_STEPS,
        "gae_lambda": GAE_LAMBDA,
        "RESOLUTION": "160x120",
        "COLOR_SPACE": "GRAY",
        "n_step": N_STEPS,
    }


def main(args):
    def wrap_env(env):
        env = CustomVizDoomWrapper(env, normalize=False, stack_frames=False, stack_size=1)
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
        args.env,
        n_envs=N_ENVS,
        wrapper_class=wrap_env,
        env_kwargs={"frame_skip": FRAME_SKIP},
    )
    
    config["RESOLUTION"] = (
        str(envs.observation_space.shape[0]) + "x" + str(envs.observation_space.shape[1])
    )
    config["COLOR_SPACE"] = "GRAY" if envs.observation_space.shape[2] == 1 else "RGB"

    
    run = wandb.init(
        project="vizdoom",
        name="vizdoom-Corridor" + str(nrModel),
        group="corridor",
        tags=[MODEL, "vizdoom", "DeadlyCorridor", config["RESOLUTION"], config["COLOR_SPACE"]],
        config=config,
        sync_tensorboard=True,
        save_code=True,
        monitor_gym=True,
    )

    if MODEL == "PPO":
        agent = PPO(
            policies.ActorCriticCnnPolicy,
            # policies.ActorCriticPolicy,
            envs,
            n_steps=N_STEPS,
            verbose=1,
            tensorboard_log=LOG_DIR + "/" + run.id,
            learning_rate=LEARNING_RATE,
            n_epochs=N_EPOCHS,
            batch_size=BATCH_SIZE,
            gamma=GAMMA,
        )
    elif MODEL == "DQN":
        agent = DQN(
            stable_baselines3.dqn.CnnPolicy,
            envs,
            learning_rate=LEARNING_RATE,
            batch_size=BATCH_SIZE,
            gamma=GAMMA,
            tensorboard_log=LOG_DIR + "/" + run.id,
            verbose=1,
            buffer_size=BUFFER_SIZE,
            train_freq=TRAIN_FREQ,
        )
    elif MODEL == "A2C":
        agent = A2C(
            policies.ActorCriticCnnPolicy,
            envs,
            learning_rate=LEARNING_RATE,
            n_steps=N_STEPS,
            gamma=GAMMA,
            gae_lambda=GAE_LAMBDA,
            verbose=1,
            tensorboard_log=LOG_DIR + "/" + run.id,
        )

    agent.learn(
        total_timesteps=TRAINING_TIMESTEPS,
        callback=WandbCallback(
            model_save_path=f"{MODEL_SAVE_DIR}/model_{nrModel}",
            verbose=1,
        ),
    )

    # agent.save(f"{MODEL_SAVE_DIR}/model_{nrModel}")
    
    run.finish()


if __name__ == "__main__":
    parser = ArgumentParser("Train stable-baselines3 PPO agents on ViZDoom.")
    parser.add_argument(
        "--env",
        default=DEFAULT_ENV,
        choices=AVAILABLE_ENVS,
        help="Name of the environment to play",
    )
    args = parser.parse_args()
    main(args)
