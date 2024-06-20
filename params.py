import sys
import os.path as osp
import torch
from thop import profile

sys.path.insert(0, osp.join(".", "main"))
sys.path.insert(0, osp.join(".", "data"))
sys.path.insert(0, osp.join(".", "common"))

from main.model import get_model
from main.config import cfg


model = get_model("test")
model.cuda()

img = torch.randn(1, 3, 512, 384).cuda()

flops, params = profile(model, inputs=({"img": img}, {}, {}, "test"))

print(f"FLOPS: {flops}, PARAMS: {params}")
