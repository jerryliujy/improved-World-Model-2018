import json
import os
from datetime import datetime
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import hydra
from omegaconf import OmegaConf
import pathlib
from hydra.utils import instantiate

from common.model_loader import load_checkpoint
from common.psnr import psnr
from datasets.dataloader import get_dataloaders
from datasets.car_racing_vision_dataset import CarRacingVisionDataset
from datasets.breakout_vision_dataset import BreakoutVisionDataset


def load_models(cfg, device):
    vision = instantiate(cfg.vision).to(device)

    vision.eval()

    if cfg.training.vision_resume:
        load_checkpoint(vision, cfg.training.vision_path)
    else:
        raise RuntimeError("Vision checkpoint is required to prepare predictor inputs.")

    return vision


def evaluate_vision(vision, test_loader, device):
    vision.eval()
    total_psnr = 0.0
    total_batches = 0

    with torch.no_grad():
        for batch in test_loader:
            images, _, _, _, _ = batch
            images = images.to(device)

            outputs, _ = vision(images)
            psnr_value = psnr(outputs, images)
            total_psnr += psnr_value.item()
            total_batches += 1
    return total_psnr / max(total_batches, 1)


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath('configs')),
    config_name='car_racing_workspace'
)
def main(cfg: OmegaConf):
    OmegaConf.resolve(cfg)

    device = torch.device(cfg.device)
    vision = load_models(cfg, device)
    dataset = instantiate(cfg.dataset)
    assert isinstance(dataset, CarRacingVisionDataset) or \
        isinstance(dataset, BreakoutVisionDataset), \
        "Dataset must be an instance of CarRacingVisionDataset or BreakoutVisionDataset."

    _, _, test_loader = get_dataloaders(
        dataset,
        batch_size=cfg.training.batch_size,
        train_ratio=cfg.dataloader.train_ratio,
        val_ratio=cfg.dataloader.val_ratio,
        test_ratio=cfg.dataloader.test_ratio,
        num_workers=cfg.dataloader.num_workers,
    )

    test_psnr = evaluate_vision(vision, test_loader, device)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    os.makedirs(cfg.eval_vision.output_dir, exist_ok=True)
    summary_path = os.path.join(cfg.eval_vision.output_dir, f"vision_test_{timestamp}.json")
    summary = {
        "config": OmegaConf.to_container(cfg, resolve=True),
        "device": str(device),
        "test_psnr": test_psnr,
        "num_test_batches": len(test_loader),
    }

    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print("Predictor evaluation complete.")
    print(json.dumps(summary, indent=2))
    print(f"Summary saved to {summary_path}")

if __name__ == "__main__":
    main()
