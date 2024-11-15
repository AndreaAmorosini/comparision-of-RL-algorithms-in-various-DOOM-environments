from stable_baselines3.common.callbacks import BaseCallback
from captum.attr import Saliency
import torch
import numpy as np
import cv2


class SaliencyCallback(BaseCallback):
    def __init__(self, model, env, saliency_freq=10, verbose=1):
        super(SaliencyCallback, self).__init__(verbose)
        self.model = model
        self.env = env
        self.saliency_freq = (
            saliency_freq  # Generate saliency every 'saliency_freq' episodes
        )
        self.saliency = Saliency(model.policy)  # Initialize saliency for model

    def _on_step(self):
        # Check if it's time to generate a saliency map
        if self.n_calls % self.saliency_freq == 0:
            obs = self.locals["new_obs"]  # Get current observation from environment
            obs_tensor = torch.tensor(obs).float().unsqueeze(0).requires_grad_(True)

            # Get model's action (without taking it in the environment)
            action_logits = self.model.policy(obs_tensor)
            action = action_logits.argmax().item()

            # Generate saliency map with respect to the chosen action
            attributions = self.saliency.attribute(obs_tensor, target=action)
            saliency_map = attributions.squeeze().cpu().detach().numpy()

            # Visualize the saliency map overlay on the original frame
            self._log_saliency_map(
                obs[0], saliency_map
            )  # Log for first observation in batch

        return True  # Continue training

    def _log_saliency_map(self, obs_frame, saliency_map):
        # Convert obs and saliency to format for visualization
        obs_frame = np.moveaxis(
            obs_frame, 0, -1
        )  # Move channels for visualization (if necessary)
        saliency_map = np.abs(saliency_map)  # Absolute values for visual clarity

        # Normalize and overlay saliency on the frame
        saliency_map = (saliency_map - saliency_map.min()) / (
            saliency_map.max() - saliency_map.min()
        )
        saliency_overlay = cv2.applyColorMap(
            (saliency_map * 255).astype(np.uint8), cv2.COLORMAP_JET
        )
        overlayed_frame = cv2.addWeighted(obs_frame, 0.6, saliency_overlay, 0.4, 0)

        # Show or save the overlay for analysis
        cv2.imshow("Saliency Map", overlayed_frame)
        cv2.waitKey(1)  # Display for a brief moment (1 ms)
