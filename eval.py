import sys
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import os
import pathlib
import hydra
import torch
import json
from datetime import datetime
from omegaconf import OmegaConf
from workspace.base_workspace import BaseWorkspace

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath('configs')),
    config_name='car_racing_workspace'
)
def main(cfg: OmegaConf):
    OmegaConf.resolve(cfg)

    output_dir = cfg.eval.output_dir
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    eval_output_dir = os.path.join(output_dir, f"eval_{timestamp}")
    pathlib.Path(eval_output_dir).mkdir(parents=True, exist_ok=True)

    cls = hydra.utils.get_class(cfg.workspace._target_)
    workspace: BaseWorkspace = cls(cfg)
    
    # get policy from workspace
    vae = workspace.vision
    predictor = workspace.predictor
    controller = workspace.controller

    device = torch.device(cfg.device)
    vae.to(device)
    predictor.to(device)
    controller.to(device)
    vae.eval()
    predictor.eval()
    controller.eval()

    # run eval
    env_runner = workspace.env_runner
    eval_results = env_runner.run(
        vae, predictor, controller, 
        num_episodes=cfg.eval.num_episodes, max_steps=cfg.eval.max_steps, render=cfg.eval.render,
        output_dir=eval_output_dir, save_video=cfg.eval.save_video
    )
    
    json_path = os.path.join(eval_output_dir, 'eval_results.json')
    json_results = {
        'timestamp': timestamp,
        'mean_reward': float(eval_results['avg_reward']),
        'std_reward': float(eval_results['std_reward']),
        'max_reward': float(max(eval_results['episode_rewards'])),
        'min_reward': float(min(eval_results['episode_rewards'])),
        'num_episodes': len(eval_results['episode_rewards']),
        'config': OmegaConf.to_container(cfg, resolve=True)
    }
    
    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2, sort_keys=True)
    print(f"JSON results saved: {json_path}")
    
    report_path = os.path.join(eval_output_dir, 'evaluation_report.txt')
    with open(report_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("EVALUATION REPORT\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Device: {cfg.device}\n")
        f.write(f"Environment: {cfg.env_runner.env_name}\n")
        f.write(f"Total Episodes: {len(eval_results['episode_rewards'])}\n")
        f.write(f"Max Steps per Episode: {cfg.eval.max_steps}\n\n")
        
        f.write("RESULTS SUMMARY\n")
        f.write("-" * 60 + "\n")
        f.write(f"Mean Reward:     {eval_results['avg_reward']:>10.2f}\n")
        f.write(f"Std Reward:      {eval_results['std_reward']:>10.2f}\n")
        f.write(f"Max Reward:      {max(eval_results['episode_rewards']):>10.2f}\n")
        f.write(f"Min Reward:      {min(eval_results['episode_rewards']):>10.2f}\n\n")
        
        f.write("EPISODE DETAILS\n")
        f.write("-" * 60 + "\n")
        f.write(f"{'Episode':<12}{'Reward':<15}{'Steps':<12}\n")
        f.write("-" * 60 + "\n")
        
        for i, (reward, steps) in enumerate(zip(
            eval_results['episode_rewards'],
            eval_results['episode_steps']
        )):
            f.write(f"{i+1:<12}{reward:<15.2f}{steps:<12}\n")
        
        f.write("\n" + "=" * 60 + "\n")
    
    print(f"Report saved: {report_path}")

    env_runner.close()

if __name__ == '__main__':
    main()