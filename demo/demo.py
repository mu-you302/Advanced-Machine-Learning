import sys
import os
import os.path as osp
import argparse
import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
from torch.nn.parallel.data_parallel import DataParallel
import torch.backends.cudnn as cudnn
import time

sys.path.insert(0, osp.join("..", "main"))
sys.path.insert(0, osp.join("..", "data"))
sys.path.insert(0, osp.join("..", "common"))
from config import cfg

# from model import get_model
from modelH import get_model
from utils.preprocessing import load_img, process_bbox, generate_patch_image
from utils.human_models import smpl_x
from utils.vis import render_mesh, save_obj, vis_3d_skeleton
import json
from torchvision import transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn


def get_one_box(det_output, thrd=0.9):
    """获取检测结果中得分最高的bbox

    Args:
        det_output (ndarray): key: boxes, scores
        thrd (float, optional): 结果阈值. Defaults to 0.9.

    Returns:
        list : 置信度最高的bbox
    """
    max_area = 0
    max_bbox = None

    if det_output["boxes"].shape[0] == 0 or thrd < 1e-5:
        return None

    for i in range(det_output["boxes"].shape[0]):
        bbox = det_output["boxes"][i]
        score = det_output["scores"][i]
        if float(score) < thrd:
            continue
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        if float(area) > max_area:
            max_bbox = [float(x) for x in bbox]
            max_area = area

    if max_bbox is None:
        return get_one_box(det_output, thrd=thrd - 0.1)

    return max_bbox


def parse_args():
    """解析命令行参数

    Returns:
        None
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, default="0", dest="gpu_ids")
    parser.add_argument("--img", type=str, default="input.png")
    args = parser.parse_args()

    # 检测GPU设置是否可用
    if not args.gpu_ids:
        assert 0, print("Please set proper gpu ids")

    if "-" in args.gpu_ids:
        gpus = args.gpu_ids.split("-")
        gpus[0] = int(gpus[0])
        gpus[1] = int(gpus[1]) + 1
        args.gpu_ids = ",".join(map(lambda x: str(x), list(range(*gpus))))

    return args


args = parse_args()
cfg.set_args(args.gpu_ids)
cudnn.benchmark = True

# 加载模型权重
model_path = "../models/snapshot_6.pth.tar"
assert osp.exists(model_path), "Cannot find model at " + model_path
print("Load checkpoint from {}".format(model_path))
model = get_model("test")
model = DataParallel(model).cuda()  # 多卡
ckpt = torch.load(model_path)  # 加载模型
model.load_state_dict(ckpt["network"], strict=False)
model.eval()  # 设置为评估模式

# 输入图像预处理
transform = transforms.ToTensor()
img_path = args.img
original_img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
original_img_height, original_img_width = original_img.shape[:2]

# 设置输出目录
output_dir = os.path.basename(img_path)
output_dir = output_dir.split(".")[0]
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

# 检测模型，采用FasterRCNN
det_model = fasterrcnn_resnet50_fpn(pretrained=True).cuda().eval()
det_transform = T.Compose([T.ToTensor()])
det_input = det_transform(original_img).cuda()
det_output = det_model([det_input])[0]
bbox = get_one_box(det_output)  # xyxy
bbox = [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]  # xywh
bbox = process_bbox(bbox, original_img_width, original_img_height)
img, img2bb_trans, bb2img_trans = generate_patch_image(
    original_img, bbox, 1.0, 0.0, False, cfg.input_img_shape
)
img = transform(img.astype(np.float32)) / 255
img = img.cuda()[None, :, :, :]

# 前向传播
inputs = {"img": img}
targets = {}
meta_info = {}
with torch.no_grad():
    out = model(inputs, targets, meta_info, "test")
# start_time = time.time()
# for i in range(100):
#     with torch.no_grad():
#         out = model(inputs, targets, meta_info, "test")
# print("Average runtime: ", (time.time() - start_time) / 100)    # 统计平均运行时间

# 获取结果中的SMPL-X模型参数
mesh = out["smplx_mesh_cam"].detach().cpu().numpy()[0]

# 保存numpy array
# with open("test.npy", "wb") as fp:
#     joint_img = out["joint_cam"].detach().cpu().numpy()[0]
#     joint_img = joint_img[:25, :]
#     left_idx = [11, 13]
#     # joint_img[left_idx, :] += 2
#     vis_3d_skeleton(joint_img, np.ones((25, 1)), None, "3d_skeleton.jpg")
#     np.save(fp, joint_img)

# 保存obj文件
# save_obj(mesh, smpl_x.face, os.path.join(output_dir, "output.obj"))

# 人体3D渲染可视化，在裁剪图像上进行
vis_img = img.cpu().numpy()[0].transpose(1, 2, 0).copy() * 255
focal = [
    cfg.focal[0] / cfg.input_body_shape[1] * cfg.input_img_shape[1],
    cfg.focal[1] / cfg.input_body_shape[0] * cfg.input_img_shape[0],
]
princpt = [
    cfg.princpt[0] / cfg.input_body_shape[1] * cfg.input_img_shape[1],
    cfg.princpt[1] / cfg.input_body_shape[0] * cfg.input_img_shape[0],
]
rendered_img = render_mesh(
    vis_img, mesh, smpl_x.face, {"focal": focal, "princpt": princpt}
)
cv2.imwrite(
    os.path.join(output_dir, "render_cropped_img.jpg"), rendered_img[:, :, ::-1]
)

# 人体3D渲染可视化，在原始图像上进行
vis_img = original_img.copy()
focal = [
    cfg.focal[0] / cfg.input_body_shape[1] * bbox[2],
    cfg.focal[1] / cfg.input_body_shape[0] * bbox[3],
]
princpt = [
    cfg.princpt[0] / cfg.input_body_shape[1] * bbox[2] + bbox[0],
    cfg.princpt[1] / cfg.input_body_shape[0] * bbox[3] + bbox[1],
]
rendered_img = render_mesh(
    vis_img, mesh, smpl_x.face, {"focal": focal, "princpt": princpt}
)
cv2.imwrite(
    os.path.join(output_dir, "render_original_img.jpg"), rendered_img[:, :, ::-1]
)

# 保存结果中的身体手部姿势和形状参数
root_pose = out["smplx_root_pose"].detach().cpu().numpy()[0]
body_pose = out["smplx_body_pose"].detach().cpu().numpy()[0]
lhand_pose = out["smplx_lhand_pose"].detach().cpu().numpy()[0]
rhand_pose = out["smplx_rhand_pose"].detach().cpu().numpy()[0]
jaw_pose = out["smplx_jaw_pose"].detach().cpu().numpy()[0]
shape = out["smplx_shape"].detach().cpu().numpy()[0]
expr = out["smplx_expr"].detach().cpu().numpy()[0]
with open(os.path.join(output_dir, "smplx_param.json"), "w") as f:
    json.dump(
        {
            "root_pose": root_pose.reshape(-1).tolist(),
            "body_pose": body_pose.reshape(-1).tolist(),
            "lhand_pose": lhand_pose.reshape(-1).tolist(),
            "rhand_pose": rhand_pose.reshape(-1).tolist(),
            "jaw_pose": jaw_pose.reshape(-1).tolist(),
            "shape": shape.reshape(-1).tolist(),
            "expr": expr.reshape(-1).tolist(),
        },
        f,
    )
