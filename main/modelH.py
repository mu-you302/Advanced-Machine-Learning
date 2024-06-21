import torch
import torch.nn as nn
from torch.nn import functional as F
from nets.resnet import ResNetBackbone
from nets.module import (
    PositionNet,
    RotationNet,
)
from nets.loss import CoordLoss, ParamLoss
from utils.human_models import mano
from utils.transforms import rot6d_to_axis_angle
from config import cfg
import math
import copy


class Model(nn.Module):
    def __init__(
        self,
        backbone,
        hand_position_net,
        hand_rotation_net,
    ):
        super(Model, self).__init__()
        self.backbone = backbone
        self.hand_position_net = hand_position_net
        self.hand_rotation_net = hand_rotation_net

        # 手部的MANO模型
        self.mano_layer = copy.deepcopy(mano.layer["right"]).cuda()

        self.coord_loss = CoordLoss()
        self.param_loss = ParamLoss()

        self.trainable_modules = [
            self.backbone,
            self.hand_position_net,
            self.hand_rotation_net,
        ]

    def get_camera_trans(self, cam_param):
        # camera translation
        t_xy = cam_param[:, :2]
        gamma = torch.sigmoid(cam_param[:, 2])  # apply sigmoid to make it positive
        k_value = (
            torch.FloatTensor(
                [
                    math.sqrt(
                        cfg.focal[0]
                        * cfg.focal[1]
                        * cfg.camera_3d_size
                        * cfg.camera_3d_size
                        / (cfg.input_body_shape[0] * cfg.input_body_shape[1])
                    )
                ]
            )
            .cuda()
            .view(-1)
        )
        t_z = k_value * gamma
        cam_trans = torch.cat((t_xy, t_z[:, None]), 1)
        return cam_trans

    def get_coord(
        self,
        root_pose,
        hand_pose,
        shape,
        cam_trans,
        mode,
    ):
        batch_size = root_pose.shape[0]
        # 通过MANO模型进行mesh重建
        output = self.mano_layer(
            betas=shape, global_orient=root_pose, hand_pose=hand_pose
        )
        # 相机坐标系
        mesh_cam = output.vertices
        joint_cam = torch.bmm(
            torch.from_numpy(mano.joint_regressor)
            .cuda()[None, :, :]
            .repeat(batch_size, 1, 1),
            mesh_cam,
        )
        root_joint_idx = mano.root_joint_idx

        # 投影到2D空间
        if (
            mode == "train"
            and len(cfg.trainset_3d) == 1
            and cfg.trainset_3d[0] == "AGORA"
            and len(cfg.trainset_2d) == 0
        ):
            x = (joint_cam[:, :, 0].detach() + cam_trans[:, None, 0]) / (
                joint_cam[:, :, 2].detach() + cam_trans[:, None, 2] + 1e-4
            ) * cfg.focal[0] + cfg.princpt[0]
            y = (joint_cam[:, :, 1].detach() + cam_trans[:, None, 1]) / (
                joint_cam[:, :, 2].detach() + cam_trans[:, None, 2] + 1e-4
            ) * cfg.focal[1] + cfg.princpt[1]
        else:
            x = (joint_cam[:, :, 0] + cam_trans[:, None, 0]) / (
                joint_cam[:, :, 2] + cam_trans[:, None, 2] + 1e-4
            ) * cfg.focal[0] + cfg.princpt[0]
            y = (joint_cam[:, :, 1] + cam_trans[:, None, 1]) / (
                joint_cam[:, :, 2] + cam_trans[:, None, 2] + 1e-4
            ) * cfg.focal[1] + cfg.princpt[1]
        x = x / cfg.input_img_shape[1] * cfg.output_hm_shape[2]
        y = y / cfg.input_img_shape[0] * cfg.output_hm_shape[1]
        joint_proj = torch.stack((x, y), 2)

        # 相对于根节点的3D坐标
        root_cam = joint_cam[:, root_joint_idx, None, :]
        joint_cam = joint_cam - root_cam

        # add camera translation for the rendering
        mesh_cam = mesh_cam + cam_trans[:, None, :]
        return joint_proj, joint_cam, mesh_cam

    def forward(self, inputs, targets, meta_info, mode):
        # backbone
        img_feat = self.backbone(inputs["img"])  # (B, 2048, 8, 6)
        # 位置估计
        joint_img = self.hand_position_net(img_feat)
        # 旋转估计
        root_pose_6d, pose_param_6d, shape_param, cam_param = self.hand_rotation_net(
            img_feat, joint_img
        )
        # change 6d pose -> axis angles
        root_pose = rot6d_to_axis_angle(root_pose_6d).reshape(-1, 3)
        pose_param = rot6d_to_axis_angle(pose_param_6d.view(-1, 6)).reshape(
            -1, (mano.orig_joint_num - 1) * 3
        )
        cam_trans = self.get_camera_trans(cam_param)

        # 获取坐标
        joint_proj, joint_cam, mesh_cam = self.get_coord(
            root_pose, pose_param, shape=shape_param, cam_trans=cam_trans, mode=mode
        )

        if mode == "train":
            loss = {}  # 计算各项损失
            loss["joint_img"] = self.coord_loss(
                joint_img,
                targets["joint_img"],
                meta_info["joint_trunc"],
                meta_info["is_3D"],
            )
            loss["mano_joint_img"] = self.coord_loss(
                joint_img, targets["mano_joint_img"], meta_info["mano_joint_trunc"]
            )
            loss["mano_pose"] = self.param_loss(
                pose_param, targets["mano_pose"], meta_info["mano_pose_valid"]
            )
            loss["mano_shape"] = self.param_loss(
                shape_param,
                targets["mano_shape"],
                meta_info["mano_shape_valid"][:, None],
            )
            loss["joint_proj"] = self.coord_loss(
                joint_proj, targets["joint_img"][:, :, :2], meta_info["joint_trunc"]
            )
            loss["joint_cam"] = self.coord_loss(
                joint_cam,
                targets["joint_cam"],
                meta_info["joint_valid"] * meta_info["is_3D"][:, None, None],
            )
            loss["mano_joint_cam"] = self.coord_loss(
                joint_cam, targets["mano_joint_cam"], meta_info["mano_joint_valid"]
            )
            return loss
        else:
            out = {}
            out["cam_trans"] = cam_trans
            out["img"] = inputs["img"]
            out["joint_img"] = joint_img
            out["mano_mesh_cam"] = mesh_cam
            out["mano_pose"] = pose_param
            out["mano_shape"] = shape_param
            if "mano_mesh_cam" in targets:
                out["mano_mesh_cam_target"] = targets["mano_mesh_cam"]
            if "joint_img" in targets:
                out["joint_img_target"] = targets["joint_img"]
            if "joint_valid" in meta_info:
                out["joint_valid"] = meta_info["joint_valid"]
            if "bb2img_trans" in meta_info:
                out["bb2img_trans"] = meta_info["bb2img_trans"]
            return out


def init_weights(m):
    try:
        if type(m) == nn.ConvTranspose2d:
            nn.init.normal_(m.weight, std=0.001)
        elif type(m) == nn.Conv2d:
            nn.init.normal_(m.weight, std=0.001)
            nn.init.constant_(m.bias, 0)
        elif type(m) == nn.BatchNorm2d:
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif type(m) == nn.Linear:
            nn.init.normal_(m.weight, std=0.01)
            nn.init.constant_(m.bias, 0)
    except AttributeError:
        pass


def get_model_hand(mode):
    backbone = ResNetBackbone(cfg.resnet_type)

    hand_position_net = PositionNet("hand", cfg.hand_resnet_type)
    hand_rotation_net = RotationNet("hand", cfg.hand_resnet_type)

    if mode == "train":
        backbone.init_weights()
        hand_position_net.apply(init_weights)
        hand_rotation_net.apply(init_weights)

    model = Model(
        backbone,
        hand_position_net,
        hand_rotation_net,
    )
    return model
