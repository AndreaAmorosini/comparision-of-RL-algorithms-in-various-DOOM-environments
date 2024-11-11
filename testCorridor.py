from argparse import ArgumentParser

import cv2
import gymnasium
import gymnasium.wrappers.human_rendering
from gymnasium.wrappers import FlattenObservation
import numpy as np
from stable_baselines3 import PPO
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
# Height and width of the resized image
IMAGE_SHAPE = (60, 80)

# Training parameters
TRAINING_TIMESTEPS = 1500000
N_STEPS = 1024
N_ENVS = 2
FRAME_SKIP = 4
LEARNING_RATE = 1e-4
N_EPOCHS = 10
BATCH_SIZE = 64
GAMMA = 0.99
CLIP_RANGE = 0.1
GAE_LAMBDA = 0.9
ENT_COEF = 0.0


CHECKPOINT_DIR = "./checkpoints/train/corridor"
LOG_DIR = "./logs/corridor"
MODEL_SAVE_DIR = "./final_models/corridor"

config = {
    "env": DEFAULT_ENV,
    "image_shape": IMAGE_SHAPE,
    "training_timesteps": TRAINING_TIMESTEPS,
    "n_steps": N_STEPS,
    "n_envs": N_ENVS,
    "frame_skip": FRAME_SKIP,
    "learning_rate": LEARNING_RATE,
    "gamma": GAMMA,
    "n_epochs": N_EPOCHS,
    "batch_size": BATCH_SIZE,
    "clip_range": CLIP_RANGE,
    "gae_lambda": GAE_LAMBDA,
    "ent_coef": ENT_COEF,
}


def main(args):
    # Create multiple environments: this speeds up training with PPO
    # We apply two wrappers on the environment:
    #  1) The above wrapper that modifies the observations (takes only the image and resizes it)
    #  2) A reward scaling wrapper. Normally the scenarios use large magnitudes for rewards (e.g., 100, -100).
    #     This may lead to unstable learning, and we scale the rewards by 1/100
    def wrap_env(env):
        # env = gymnasium.make("VizdoomBasic-v0", render_mode="human", frame_skip=4)
        env = CustomVizDoomWrapper(env, normalize=True, stack_frames=False, stack_size=1)
        env = gymnasium.wrappers.TransformReward(env, lambda r: r * 0.01)
        # env = ObservationWrapper(env)
        # env = FlattenObservation(env)
        env = gymnasium.wrappers.HumanRendering(env)
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
    
    run = wandb.init(
        project="vizdoom",
        name="vizdoom-Corridor" + str(nrModel),
        group="corridor",
        tags=["ppo", "vizdoom", "DeadlyCorridor"],
        config=config,
        sync_tensorboard=True,
        save_code=True,
        monitor_gym=True,
    )


    agent = PPO(
        policies.MultiInputActorCriticPolicy,
        envs,
        n_steps=N_STEPS,
        verbose=2,
        tensorboard_log=LOG_DIR + "/" + run.id,
        learning_rate=LEARNING_RATE,
        n_epochs=N_EPOCHS,
        batch_size=BATCH_SIZE,
        gamma=GAMMA,
        clip_range=CLIP_RANGE,
        gae_lambda=GAE_LAMBDA,
        ent_coef=ENT_COEF,
    )

    # Do the actual learning
    # This will print out the results in the console.
    # If agent gets better, "ep_rew_mean" should increase steadily
    agent.learn(
        total_timesteps=TRAINING_TIMESTEPS,
        callback=WandbCallback(
            model_save_path=f"{MODEL_SAVE_DIR}/model_{nrModel}",
            verbose=2,
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
