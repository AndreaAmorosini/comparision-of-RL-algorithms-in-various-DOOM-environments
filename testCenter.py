from argparse import ArgumentParser

import cv2
import gymnasium
import gymnasium.wrappers.human_rendering
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common import evaluation, policies
import os
import wandb
from wandb.integration.sb3 import WandbCallback
from customWrapper import CustomVizDoomWrapper, ObservationWrapper


import vizdoom.gymnasium_wrapper  # noqa


# DEFAULT_ENV = "VizdoomDefendCenter-v0"
DEFAULT_ENV = "VizdoomDefendCenter-custom-v0"
AVAILABLE_ENVS = [env for env in gymnasium.envs.registry.keys() if "Vizdoom" in env]
# Height and width of the resized image
IMAGE_SHAPE = (60, 80)
EVAL_FREQ = 50000

# Training parameters
TRAINING_TIMESTEPS = 1500000
N_STEPS = 2048
N_ENVS = 1
FRAME_SKIP = 4
LEARNING_RATE = 1e-4
N_EPOCHS = 10
BATCH_SIZE = 64
GAMMA = 0.99
CLIP_RANGE = 0.2
GAE_LAMBDA = 0.95
ENT_COEF = 0.0

CHECKPOINT_DIR = "./checkpoints/train/center"
LOG_DIR = "./logs/center"
MODEL_SAVE_DIR = "./final_models/center"

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
    "RESOLUTION": "320x240",
    "COLOR_SPACE": "RGB",
}


def main(args):
    def wrap_env(env):
        env = CustomVizDoomWrapper(env, normalize=False, stack_frames=False, stack_size=1)
        # env = gymnasium.wrappers.TransformReward(env, lambda r: r / 1000)
        env = gymnasium.wrappers.HumanRendering(env)
        return env

    models = os.listdir(MODEL_SAVE_DIR)

    if models == []:
        nrModel = 0
    else:
        nrModel = len(models) + 1

    envs = make_vec_env(args.env, n_envs=N_ENVS, wrapper_class=wrap_env, env_kwargs={"frame_skip": FRAME_SKIP})

    # Initialize wandb
    run = wandb.init(
        project="vizdoom",
        name="vizdoom_Center_" + str(nrModel),
        group="center",
        tags=["ppo", "vizdoom", "DefendCenter", config["RESOLUTION"], config["COLOR_SPACE"]],
        config=config,
        sync_tensorboard=True,
        save_code=True,
        monitor_gym=True,
    )

    agent = PPO(
        policies.ActorCriticCnnPolicy, envs,
        n_steps=N_STEPS,
        verbose=1,
        tensorboard_log=LOG_DIR + "/" + run.id,
        learning_rate=LEARNING_RATE, 
        n_epochs=N_EPOCHS,
        batch_size=BATCH_SIZE,
        gamma=GAMMA,
    )
    
    # agent = PPO(
    #     policies.MultiInputActorCriticPolicy, envs, n_steps=N_STEPS, verbose=1, tensorboard_log=LOG_DIR + "/" + str(nrModel),
    #     learning_rate=1e-5, 
    #     n_epochs=10,
    #     batch_size=64,
    # )



    agent.learn(
        total_timesteps=TRAINING_TIMESTEPS,
        callback=WandbCallback(
            model_save_path=f"{MODEL_SAVE_DIR}/model_{nrModel}",
            verbose=1,
        ),
    )
    
    # run.log_model(path=f"{MODEL_SAVE_DIR}/model_{nrModel}/model.zip", name="vizDoom_Center_" + str(nrModel))
    
    run.finish()
    
    # agent.save(f"{MODEL_SAVE_DIR}/model_{nrModel}")


if __name__ == "__main__":
    parser = ArgumentParser("Train stable-baselines3 PPO agents on ViZDoom.")
    parser.add_argument(
        "--env",
        default=DEFAULT_ENV,
        choices=AVAILABLE_ENVS,
        help="Name of the environment to play",
    )
    args = parser.parse_args()
    
    try:
        main(args)
    except KeyboardInterrupt:
        print("Training interrupted")
        # run.finish()
