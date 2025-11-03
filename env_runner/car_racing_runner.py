import gymnasium as gym
import torch
import cv2
import numpy as np
from common.img_process import preprocess_carracing_image


class CarRacingRunner:
    def __init__(
        self,
        env_name='CarRacing-v3',
        save_video=False,
        video_filename='car_racing_eval.mp4',
        resolution=(96, 96),
        device: str = None,
        to_grayscale: bool = False,
    ):
        self.env_name = env_name
        self.save_video = save_video
        self.video_filename = video_filename
        self.resolution = resolution
        self.to_grayscale = to_grayscale

        # determine device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.env = gym.make(self.env_name, render_mode='rgb_array')

    def run(self, vision, predictor, controller, num_episodes: int = 1, max_steps: int = 1000, render: bool = False):
        """
        Run evaluation rollouts using the provided models.

        Args:
            vision: vision model (VAE / encoder) - provides encode(img) -> z
            predictor: memory model (MDNRNN) - provides forward(z, a, h) -> (output, h_new)
            controller: controller model - provides get_action(state) -> action
            num_episodes: number of episodes to run
            max_steps: max steps per episode
            render: whether to call env.render() each step
        """
        env = self.env

        # Move models to runner device and set eval mode
        vision.to(self.device)
        predictor.to(self.device)
        controller.to(self.device)
        vision.eval()
        predictor.eval()
        controller.eval()

        for ep in range(num_episodes):
            obs, _ = env.reset()
            cumulative_reward = 0.0

            # setup video writer if requested
            video_writer = None
            if self.save_video:
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                video_writer = cv2.VideoWriter(
                    self.video_filename, fourcc, 30.0, (self.resolution[0], self.resolution[1])
                )

            # Initialize hidden state (h is None, predictor handles initialization)
            h = None
            step_count = 0

            while True:
                # Preprocess observation to tensor
                img_t = preprocess_carracing_image(obs, to_grayscale=self.to_grayscale)
                img_t = img_t.unsqueeze(0).to(self.device)  # (1, C, H, W)

                with torch.no_grad():
                    # Encode: vision returns z directly
                    z = vision.encode(img_t)  # (1, latent_dim)

                    # Prepare z for predictor (add sequence dimension)
                    z_seq = z.unsqueeze(1)  # (1, 1, latent_dim)

                    # Get action: controller takes (z + h_state)
                    # h[0] is (num_layers, batch, hidden_dim), take last layer
                    if h is not None:
                        h_state = h[0][-1]  # (batch, hidden_dim)
                    else:
                        # First step: h_state is zeros
                        hidden_dim = predictor.hidden_dim
                        h_state = torch.zeros(1, hidden_dim, device=self.device)
                    
                    state = torch.cat([z.squeeze(0), h_state.squeeze(0)], dim=-1)  # (state_dim,)
                    a = controller.get_action(state)  # (action_dim,)

                    # Convert action to numpy for env.step
                    step_action = a.detach().cpu().numpy()

                    # Update predictor hidden state: forward(z, a, h) -> h_new
                    a_seq = torch.tensor(step_action, dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)
                    _, h = predictor(z_seq, a_seq, h=h)

                # Step environment
                obs, reward, done, truncated, info = env.step(step_action)
                if render:
                    env.render()

                # Write video frame if needed
                if video_writer is not None:
                    frame = cv2.resize(obs, self.resolution)
                    video_writer.write(frame[:, :, ::-1])  # RGB->BGR for OpenCV

                cumulative_reward += reward
                step_count += 1

                if done or truncated or step_count >= max_steps:
                    break

            if video_writer is not None:
                video_writer.release()

            print(f'Episode {ep+1} | Reward: {cumulative_reward:.2f} | Steps: {step_count}')

        env.close()