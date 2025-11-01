import gymnasium as gym
import torch
import cv2
import numpy as np
from env_runner.base_runner import BaseRunner

class CarRacingRunner(BaseRunner):
    def __init__(self, 
                 env_name='CarRacing-v3',
                 save_video=False,
                 video_filename='car_racing_eval.mp4',
                 resolution=(96, 96)):
        super().__init__()
        self.env_name = env_name
        self.save_video = save_video
        self.video_filename = video_filename
        self.resolution = resolution

        self.env = gym.make(self.env_name, render_mode='rgb_array')
        
    def run(self, vision, predictor, controller):
        env = self.env
        done = False
        cumulative_reward = 0.0
        obs, _ = env.reset()
        
        video_writer = None
        if self.save_video:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            video_writer = cv2.VideoWriter(self.video_filename, fourcc, 30.0, (self.resolution[0], self.resolution[1]))

        # Initialize hidden state
        h = (torch.zeros(1, HIDDEN_DIM).to(self.device),
            torch.zeros(1, HIDDEN_DIM).to(self.device))
        step_count = 1
        
        while True:
            # Encode observation to latent space
            z = vision.encode(obs[np.newaxis, ...])

            # Combine latent and hidden state
            x = torch.cat([z, h[0]], dim=-1)
            
            # Get action from controller
            a = controller.get_action(x)
            
            # Step environment
            obs, reward, done, _, _ = env.step(a.detach().cpu().numpy())
            env.render()
            
            # Update LSTM hidden state
            _, h = predictor.rnn(z, a.unsqueeze(0), h=h)
        
            cumulative_reward += reward
            step_count += 1
            
            # End episode on completion or timeout
            if done or step_count >= 1000:
                break
        
        env.close()
        print(f'Reward: {cumulative_reward:.2f} | Steps: {step_count}')