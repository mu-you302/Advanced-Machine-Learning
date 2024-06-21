import torch
import os.path as osp

model_path = osp.join(".", "ckpt_6.pth.tar")
model = torch.load(model_path)
model["epoch"] = 0

save_path = osp.join(".", "ckpt_0.pth.tar")
torch.save(model, save_path)
