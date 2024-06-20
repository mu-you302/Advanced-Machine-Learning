# from pycocotools.coco import COCO

# db = COCO(
#     "/home/ubuntu/xiyang/data/data_xiyang/Hand4Whole_RELEASE/data/PW3D/data/3DPW_test.json"
# )

# for k in db.anns.keys():
#     ann = db.anns[k]
#     image_id = ann["image_id"]
#     img = db.loadImgs(image_id)[0]
#     sequence_name = img["sequence"]
#     img_name = img["file_name"]

#     if sequence_name == "downtown_enterShop_00" and img_name == "image_00432.jpg":
#         smpl_param = ann["smpl_param"]
#         # 将这个文件保存为一个json文件
#         # import json

#         # with open("temp.json", "w") as f:
#         #     json.dump(smpl_param, f)
#         print(img["cam_param"])


import sys
import os.path as osp

sys.path.insert(0, "main")
# sys.path.insert(0, osp.join("..", "data"))
sys.path.insert(0, "common")
from utils.human_models import smpl
from utils.vis import render_mesh

smpl_model = smpl.layer["male"]

# 加载json文件
import json

with open("temp.json", "r") as f:
    smpl_param = json.load(f)

import torch
import numpy as np
import cv2

# get smpl parameters
shape = smpl_param["shape"]
pose = smpl_param["pose"]
trans = smpl_param["trans"]
shape = torch.tensor(shape).view(1, -1).float()
pose = torch.tensor(pose).view(-1, 3).float()
trans = torch.tensor(trans).view(1, -1).float()

root_pose = pose[smpl.orig_root_joint_idx].view(1, 3)

root_pose = pose[smpl.orig_root_joint_idx].view(1, 3)
body_pose = torch.cat(
    (
        pose[: smpl.orig_root_joint_idx, :],
        pose[smpl.orig_root_joint_idx + 1 :, :],
    )
).view(1, -1)
with torch.no_grad():
    output = smpl_model(
        betas=shape, body_pose=body_pose, global_orient=root_pose, transl=trans
    )
mesh_cam = output.vertices[0].numpy()
joint_cam = np.dot(smpl.joint_regressor, mesh_cam)

img_path = osp.join(
    "/home/ubuntu/xiyang/data/data_xiyang/Hand4Whole_RELEASE/data/PW3D",
    "imageFiles",
    "downtown_enterShop_00",
    "image_00432.jpg",
)
original_img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
focal = [1961.8528610354224, 1969.2307692307693]
princpt = [540.0, 960.0]
rendered_img = render_mesh(
    original_img, mesh_cam, smpl.face, {"focal": focal, "princpt": princpt}
)
cv2.imwrite("render_original_img.jpg", rendered_img[:, :, ::-1])
