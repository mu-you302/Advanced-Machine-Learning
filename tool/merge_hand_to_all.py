import torch

model_all = torch.load("../output/model_dump/ckpt_6_all.pth.tar")  # 全身模型
model_hand = torch.load("../output/model_dump/ckpt_9_hand.pth.tar")  # 手部模型

dump_keys = []
for k, v in model_hand["network"].items():
    if "module.backbone" in k:
        _k = k.split("module.backbone.")[1]
        save_k = "module.hand_roi_net.backbone." + _k
        model_all["network"][save_k] = v.cpu()

# 舍弃手部position和rotation的网络
dump_keys = []
for k in model_all["network"].keys():
    if "hand_position_net" in k or "hand_rotation_net" in k:
        dump_keys.append(k)
for k in dump_keys:
    model_all["network"].pop(k)

model_all["epoch"] = 0
torch.save(model_all, "ckpt_0.pth.tar")
