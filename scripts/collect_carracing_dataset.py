import gymnasium as gym
import numpy as np
import h5py
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import gc 
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.img_process import resize_obs

def random_action(t):
    """
    Generate pseudo-random actions based on the current time step (t).
    
    Args:
        t: Current time step in the episode
        
    Returns:
        numpy.ndarray: Action vector [steering, acceleration, brake]
    """
    if t < 20:
        return np.array([-.1, 1, 0])

    actions = [
        np.array([0, np.random.random(), 0]),    # Random Accelerate
        np.array([-np.random.random(), 0, 0]),   # Random Turn Left
        np.array([np.random.random(), 0, 0]),    # Random Turn Right
        np.array([0, 0, np.random.random()]),    # Random Brake
    ]
    probabilities = [.35, .3, .3, .05]  # Probabilities for each action

    # Select a random action based on the defined probabilities
    selected_action = np.random.choice(len(actions), p=probabilities)
    return actions[selected_action]



def collect_episode(env_name, max_steps, episode_id, worker_id, output_dir):
    """
    Collect data for a single episode and write it to an HDF5 file.
    
    Args:
        env_name (str): The Gym environment identifier
        max_steps (int): Maximum number of steps per episode
        episode_id (int): The episode's unique identifier
        worker_id (int): The worker's unique identifier
        output_dir (str): Directory to store episode files
    """
    # Initialize lists to store episode data
    episode_images = []
    episode_actions = []
    episode_rewards = []
    episode_dones = []
    
    env = gym.make(env_name, render_mode='rgb_array')
    obs, _ = env.reset()

    for step in range(max_steps):
        # Process and store the observation
        resized_image = resize_obs(obs)
        episode_images.append(resized_image)

        # Generate action (new pseudo-random action every 20 steps)
        if step % 20 == 0:
            action = random_action(step)
        episode_actions.append(action)

        # Execute action and store results
        obs, reward, done, truncated, info = env.step(action)
        episode_rewards.append(reward)
        episode_dones.append(done)

        # Reset if episode terminates early
        if done:
            obs, _ = env.reset()

    # Convert lists to NumPy arrays
    episode_images = np.array(episode_images, dtype=np.uint8)
    episode_actions = np.array(episode_actions, dtype=np.float32)
    episode_rewards = np.array(episode_rewards, dtype=np.float32)
    episode_dones = np.array(episode_dones, dtype=bool)

    # Define the episode file path
    episode_filename = f'worker_{worker_id}_episode_{episode_id}.h5'
    episode_path = os.path.join(output_dir, episode_filename)

    # Write the episode data to an HDF5 file
    with h5py.File(episode_path, 'w') as h5f:
        h5f.create_dataset('images', data=episode_images, dtype='uint8')
        h5f.create_dataset('actions', data=episode_actions, dtype='float32')
        h5f.create_dataset('rewards', data=episode_rewards, dtype='float32')
        h5f.create_dataset('dones', data=episode_dones, dtype='bool')

    # Clean up memory
    env.close()
    del episode_images, episode_actions, episode_rewards, episode_dones
    del resized_image, action, reward, done, truncated, info, obs, env
    gc.collect()  # Explicitly trigger garbage collection
    

def collect_data_worker(args):
    """
    Worker function to collect multiple episodes and write each to a separate file.
    
    Args:
        args: Tuple containing (env_name, num_episodes, max_steps, output_dir, worker_id)
    """
    env_name, num_episodes, max_steps, output_dir, worker_id = args

    for ep in range(num_episodes):
        collect_episode(env_name, max_steps, ep, worker_id, output_dir)
        print(f"Worker {worker_id}: Episode ({ep+1}/{num_episodes}) completed")

    print(f"Worker {worker_id}: Completed {num_episodes} episodes.")


def collect_data(env_name='CarRacing-v3', num_episodes=100, max_steps=1000, 
                output_dir='episodes_data', num_workers=None):
    """
    Collect data from the environment and store each episode in a separate HDF5 file.
    
    Args:
        env_name (str): Name of the Gym environment
        num_episodes (int): Total number of episodes to generate
        max_steps (int): Maximum steps per episode
        output_dir (str): Directory to store episode files
        num_workers (int, optional): Number of worker processes. Defaults to CPU count
    """
    if num_workers is None:
        num_workers = cpu_count()-1 # Use all available CPU cores

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Determine episodes per worker
    episodes_per_worker = num_episodes // num_workers
    remaining_episodes = num_episodes % num_workers

    # Prepare arguments for each worker
    worker_args = []
    for i in range(num_workers):
        worker_episodes = episodes_per_worker + (1 if i < remaining_episodes else 0)
        args = (env_name, worker_episodes, max_steps, output_dir, i)
        worker_args.append(args)

    # Use Pool for multiprocessing with controlled number of workers
    with Pool(processes=num_workers) as pool:
        list(pool.imap_unordered(collect_data_worker, worker_args))

    print("Data collection completed.")
    

def merge_episode_files(episodes_dir='temp_episodes_data',
                        output_file='merged_data.h5', 
                        env_name='CarRacing-v3',
                        num_max_steps=1000):
    """
    Merge multiple episode HDF5 files into a single HDF5 dataset.
    
    Args:
        episodes_dir (str): Directory containing individual episode files
        output_file (str): Path for the consolidated output file
        env_name (str): Environment name (for documentation purposes)
        num_max_steps (int): Expected number of steps per episode
    """
    # Step 1: Locate all episode files
    episode_files = [
        os.path.join(episodes_dir, f)
        for f in os.listdir(episodes_dir)
        if f.endswith('.h5') or f.endswith('.hdf5')
    ]

    num_episodes = len(episode_files)
    if num_episodes == 0:
        raise ValueError(f"No HDF5 episode files found in directory: {episodes_dir}")

    print(f"Found {num_episodes} episode files in '{episodes_dir}'.")

    # Optional: Sort files for consistent ordering
    episode_files.sort()

    # Step 2: Verify episode integrity
    for file in episode_files:
        with h5py.File(file, 'r') as h5f:
            actions_shape = h5f['actions'].shape
            if actions_shape[0] != num_max_steps:
                raise ValueError(
                    f"Episode file {file} has {actions_shape[0]} steps, expected {num_max_steps}."
                )

    # Step 3: Initialize the merged HDF5 file
    with h5py.File(output_file, 'w') as merged_h5f:
        # Initialize datasets with optimized chunking
        actions_shape = (num_episodes, num_max_steps, 3)
        dones_shape = (num_episodes, num_max_steps)
        images_shape = (num_episodes, num_max_steps, 96, 96, 3)
        rewards_shape = (num_episodes, num_max_steps)

        print("Initializing datasets in the merged HDF5 file...")
        merged_h5f.create_dataset('actions', shape=actions_shape,
                                 dtype='float16', chunks=(1, num_max_steps, 3))
        merged_h5f.create_dataset('dones', shape=dones_shape,
                                 dtype='bool', chunks=(1, num_max_steps))
        merged_h5f.create_dataset('images', shape=images_shape,
                                 dtype='uint8', chunks=(1, num_max_steps, 96, 96, 3))
        merged_h5f.create_dataset('rewards', shape=rewards_shape,
                                 dtype='int', chunks=(1, num_max_steps))

        # Step 4: Iterate through each episode and write to the merged file
        for idx, episode_file in enumerate(tqdm(episode_files, desc="Merging Episodes")):
            with h5py.File(episode_file, 'r') as ep_h5f:
                # Read datasets from the episode file
                actions = ep_h5f['actions'][:]
                dones = ep_h5f['dones'][:]
                images = ep_h5f['images'][:]
                rewards = ep_h5f['rewards'][:]

                # Write to the merged datasets
                merged_h5f['actions'][idx, :, :] = actions
                merged_h5f['dones'][idx, :] = dones
                merged_h5f['images'][idx, :, :, :, :] = images
                merged_h5f['rewards'][idx, :] = rewards

            # Step 5: Clean up temporary files
            try:
                os.remove(episode_file)
            except Exception as de:
                print(f"Error deleting temporary file '{episode_file}': {de}")

    print(f"Merging completed successfully. Merged data saved to '{output_file}'.")
    os.rmdir(episodes_dir)
    
    
    
if __name__ == "__main__":
    collect_data(env_name='CarRacing-v3', num_episodes=192, max_steps=1000, 
                 output_dir='temp_episodes_data', num_workers=8)
    
    merge_episode_files(episodes_dir='temp_episodes_data',
                        output_file='outputs/data/car_racing_data.h5', 
                        env_name='CarRacing-v3',
                        num_max_steps=1000)