import os
import torch
import torchvision


def save_results(images: torch.Tensor, gt: torch.Tensor, savedir, save_name = 'prediction.png'):
    vis = torch.cat([images, gt], dim = 0)
    grid = torchvision.utils.make_grid(vis, nrow=16, normalize=True)
    grid = grid.mul(255).add_(0.5).clamp_(0, 255).to(torch.uint8).cpu()
    os.makedirs(savedir, exist_ok = True)
    torchvision.io.write_png(grid, os.path.join(savedir, save_name))