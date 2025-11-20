import torch

def psnr(img_pred: torch.Tensor, img_gt: torch.Tensor, max_pixel: float = 1.0) -> torch.Tensor:
    # img_* shape: (N, C, H, W) or (C, H, W), values in [0, max_pixel]
    mse = torch.mean((img_pred - img_gt) ** 2)
    if mse.item() == 0:
        return torch.tensor(float('inf'))
    return 10 * torch.log10((max_pixel ** 2) / mse)