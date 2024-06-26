import torch
import torch.nn as nn
from torch.nn import functional as F
from nets.resnet import ResNetBackbone
from nets.module import (
    PositionNet,
    RotationNet,
    FaceRegressor,
    BoxNet,
    HandRoI,
    FaceRoI,
)
from nets.loss import CoordLoss, ParamLoss
from utils.human_models import smpl_x
from utils.transforms import rot6d_to_axis_angle, restore_bbox
from config import cfg
import math
import copy


# 构建模型的类别，包括主干、位置网络、旋转网络、框网络、手部 ROI 网络和面部 ROI 网络
# 完成了模型的初始化、前向传播和损失函数的计算
class Model(nn.Module):
    def __init__(
        self,
        backbone,
        body_position_net,
        body_rotation_net,
        box_net,
        hand_roi_net,
        hand_position_net,
        hand_rotation_net,
        face_roi_net,
        face_regressor,
    ):
        super(Model, self).__init__()
        self.backbone = backbone
        self.body_position_net = body_position_net
        self.body_rotation_net = body_rotation_net
        self.box_net = box_net

        self.hand_roi_net = hand_roi_net
        self.hand_position_net = hand_position_net
        self.hand_rotation_net = hand_rotation_net

        self.face_roi_net = face_roi_net
        self.face_regressor = face_regressor

        self.smplx_layer = copy.deepcopy(smpl_x.layer["neutral"]).cuda()  # SMPLX 模型

        self.coord_loss = CoordLoss()
        self.param_loss = ParamLoss()

        self.trainable_modules = [
            self.backbone,
            self.body_position_net,
            self.body_rotation_net,
            self.box_net,
            self.hand_roi_net,
            self.hand_position_net,
            self.hand_rotation_net,
            self.face_roi_net,
            self.face_regressor,
        ]

    def get_camera_trans(self, cam_param):
        """
        根据给定的相机参数计算相机平移

        Args:
          cam_param: 相机参数，包括 x-y 平移和 z 平移。

        Returns:
          返回相机平移
        """
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
        body_pose,
        lhand_pose,
        rhand_pose,
        jaw_pose,
        shape,
        expr,
        cam_trans,
        mode,
    ):
        """
        以各种姿势和参数作为输入，处理它们以获得3D坐标，并返回投影2D坐标、根相对3D坐标和网格坐标。

        Args:
          root_pose: 表示根在3D空间中的姿势，包括躯干、手臂、腿等身体部位的旋转和方向
          body_pose: 表示身体在3D空间中的姿势
          lhand_pose: 表示左手姿势的参数
          rhand_pose: 输入数据中右手的姿势
          jaw_pose: 下颌在 3D 空间中的姿势
          shape:  SMPL-X 模型中使用的身体形状参数
          expr: 以张量的形式表示人脸的表情
          cam_trans: 表示相机在 3D 空间中的平移 用于调整相机相对于场景或被捕获物体的 3D 坐标的位置
          mode: 确定操作模式

        Returns:
          返回三个张量：“joint_proj”、“joint_cam”和“mesh_cam”
        """
        batch_size = root_pose.shape[0]
        zero_pose = (
            torch.zeros((1, 3)).float().cuda().repeat(batch_size, 1)
        )  # eye poses
        # output = self.smplx_layer(
        #     betas=shape,
        #     body_pose=body_pose,
        #     global_orient=root_pose,
        #     right_hand_pose=rhand_pose,
        #     left_hand_pose=lhand_pose,
        #     jaw_pose=jaw_pose,
        #     leye_pose=zero_pose,
        #     reye_pose=zero_pose,
        #     expression=expr,
        # )
        output = self.smplx_layer(
            betas=shape,
            body_pose=body_pose,
            global_orient=root_pose,
            right_hand_pose=rhand_pose,
            left_hand_pose=lhand_pose,
            jaw_pose=torch.zeros((1, 3)).float().cuda().repeat(batch_size, 1),
            leye_pose=zero_pose,
            reye_pose=zero_pose,
            expression=torch.zeros((1, 10)).float().cuda().repeat(batch_size, 1),
        )  # 生成 SMPLX 模型的输出
        # 相机中心的 3D 坐标
        mesh_cam = output.vertices
        if (
            mode == "test" and cfg.testset == "AGORA"
        ):  # use 144 joints for AGORA evaluation
            joint_cam = output.joints
        else:
            joint_cam = output.joints[:, smpl_x.joint_idx, :]  # (B, 137, 3)

        # 投影 3D 坐标到 2D 空间
        if (
            mode == "train"
            and len(cfg.trainset_3d) == 1
            and cfg.trainset_3d[0] == "AGORA"
            and len(cfg.trainset_2d) == 0
        ):  # prevent gradients from backpropagating to SMPLX paraemter regression module
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
        x = x / cfg.input_body_shape[1] * cfg.output_hm_shape[2]
        y = y / cfg.input_body_shape[0] * cfg.output_hm_shape[1]
        joint_proj = torch.stack((x, y), 2)

        # 相对于根节点的 3D 坐标
        root_cam = joint_cam[:, smpl_x.root_joint_idx, None, :]
        joint_cam = joint_cam - root_cam
        mesh_cam = mesh_cam + cam_trans[:, None, :]  # for rendering

        # 左手根（左腕）相对于 3D 坐标
        lhand_idx = smpl_x.joint_part["lhand"]
        lhand_cam = joint_cam[:, lhand_idx, :]
        lwrist_cam = joint_cam[:, smpl_x.lwrist_idx, None, :]
        lhand_cam = lhand_cam - lwrist_cam
        joint_cam = torch.cat(
            (
                joint_cam[:, : lhand_idx[0], :],
                lhand_cam,
                joint_cam[:, lhand_idx[-1] + 1 :, :],
            ),
            1,
        )

        # 右手根（右腕）相对于 3D 坐标
        rhand_idx = smpl_x.joint_part["rhand"]
        rhand_cam = joint_cam[:, rhand_idx, :]
        rwrist_cam = joint_cam[:, smpl_x.rwrist_idx, None, :]
        rhand_cam = rhand_cam - rwrist_cam
        joint_cam = torch.cat(
            (
                joint_cam[:, : rhand_idx[0], :],
                rhand_cam,
                joint_cam[:, rhand_idx[-1] + 1 :, :],
            ),
            1,
        )

        # 脸部根（颈部）相对于 3D 坐标
        face_idx = smpl_x.joint_part["face"]
        face_cam = joint_cam[:, face_idx, :]
        neck_cam = joint_cam[:, smpl_x.neck_idx, None, :]
        face_cam = face_cam - neck_cam
        joint_cam = torch.cat(
            (
                joint_cam[:, : face_idx[0], :],
                face_cam,
                joint_cam[:, face_idx[-1] + 1 :, :],
            ),
            1,
        )

        return joint_proj, joint_cam, mesh_cam

    def forward(self, inputs, targets, meta_info, mode):
        # backbone
        body_img = F.interpolate(
            inputs["img"], cfg.input_body_shape
        )  # (B ,3, 256, 192)
        img_feat = self.backbone(body_img)  # (B, 2048, 8, 6)

        # body
        body_joint_hm, body_joint_img = self.body_position_net(img_feat)

        # 手部/面部边界框和特征提取
        (
            lhand_bbox_center,
            lhand_bbox_size,
            rhand_bbox_center,
            rhand_bbox_size,
            face_bbox_center,
            face_bbox_size,
        ) = self.box_net(img_feat, body_joint_hm.detach(), body_joint_img.detach())
        lhand_bbox = restore_bbox(
            lhand_bbox_center,
            lhand_bbox_size,
            cfg.input_hand_shape[1] / cfg.input_hand_shape[0],
            2.0,
        ).detach()  # xyxy in (cfg.input_body_shape[1], cfg.input_body_shape[0]) space
        rhand_bbox = restore_bbox(
            rhand_bbox_center,
            rhand_bbox_size,
            cfg.input_hand_shape[1] / cfg.input_hand_shape[0],
            2.0,
        ).detach()  # xyxy in (cfg.input_body_shape[1], cfg.input_body_shape[0]) space
        face_bbox = restore_bbox(
            face_bbox_center,
            face_bbox_size,
            cfg.input_face_shape[1] / cfg.input_face_shape[0],
            1.5,
        ).detach()  # xyxy in (cfg.input_body_shape[1], cfg.input_body_shape[0]) space
        hand_feat = self.hand_roi_net(
            inputs["img"], lhand_bbox, rhand_bbox
        )  # hand_feat: flipped left hand + right hand
        face_feat = self.face_roi_net(inputs["img"], face_bbox)

        # hand
        hand_joint_hm, hand_joint_img = self.hand_position_net(
            hand_feat
        )  # (2N, J_P, 3)
        hand_pose = self.hand_rotation_net(
            hand_feat, hand_joint_img.detach(), hand_joint_hm
        )
        hand_pose = rot6d_to_axis_angle(hand_pose.reshape(-1, 6)).reshape(
            hand_feat.shape[0], -1
        )  # (2N, J_R*3)
        # restore flipped left hand joint coordinates
        batch_size = hand_joint_img.shape[0] // 2
        lhand_joint_img = hand_joint_img[:batch_size, :, :]
        lhand_joint_img = torch.cat(
            (
                cfg.output_hand_hm_shape[2] - 1 - lhand_joint_img[:, :, 0:1],
                lhand_joint_img[:, :, 1:],
            ),
            2,
        )
        rhand_joint_img = hand_joint_img[batch_size:, :, :]
        # 恢复翻转的左手关节旋转
        batch_size = hand_pose.shape[0] // 2
        lhand_pose = hand_pose[:batch_size, :].reshape(
            -1, len(smpl_x.orig_joint_part["lhand"]), 3
        )
        lhand_pose = torch.cat((lhand_pose[:, :, 0:1], -lhand_pose[:, :, 1:3]), 2).view(
            batch_size, -1
        )
        rhand_pose = hand_pose[batch_size:, :]
        # 恢复翻转的左手特征
        batch_size = hand_feat.shape[0] // 2
        lhand_feat = torch.flip(hand_feat[:batch_size, :], [3])
        rhand_feat = hand_feat[batch_size:, :]

        # body
        root_pose, body_pose, shape, cam_param = self.body_rotation_net(
            img_feat,
            body_joint_img.detach(),
            lhand_feat,
            lhand_joint_img[:, smpl_x.pos_joint_part["L_MCP"], :].detach(),
            rhand_feat,
            rhand_joint_img[:, smpl_x.pos_joint_part["R_MCP"], :].detach(),
        )
        root_pose = rot6d_to_axis_angle(root_pose)
        body_pose = rot6d_to_axis_angle(body_pose.reshape(-1, 6)).reshape(
            body_pose.shape[0], -1
        )  # (N, J_R*3)
        cam_trans = self.get_camera_trans(cam_param)

        # face
        expr, jaw_pose = self.face_regressor(face_feat)
        jaw_pose = rot6d_to_axis_angle(jaw_pose)

        # final output
        joint_proj, joint_cam, mesh_cam = self.get_coord(
            root_pose,
            body_pose,
            lhand_pose,
            rhand_pose,
            jaw_pose,
            shape,
            expr,
            cam_trans,
            mode,
        )
        pose = torch.cat((root_pose, body_pose, lhand_pose, rhand_pose, jaw_pose), 1)
        joint_img = torch.cat(
            (body_joint_img, lhand_joint_img, rhand_joint_img), 1
        )  # (B, J, 3)

        if mode == "train":
            # 计算损失
            loss = {}
            loss["smplx_pose"] = self.param_loss(
                pose, targets["smplx_pose"], meta_info["smplx_pose_valid"]
            )
            loss["smplx_shape"] = self.param_loss(
                shape, targets["smplx_shape"], meta_info["smplx_shape_valid"][:, None]
            )
            # loss["smplx_expr"] = self.param_loss(
            #     expr, targets["smplx_expr"], meta_info["smplx_expr_valid"][:, None]
            # )
            loss["joint_cam"] = self.coord_loss(
                joint_cam,
                targets["joint_cam"],
                meta_info["joint_valid"] * meta_info["is_3D"][:, None, None],
            )
            loss["smplx_joint_cam"] = self.coord_loss(
                joint_cam, targets["smplx_joint_cam"], meta_info["smplx_joint_valid"]
            )
            loss["lhand_bbox"] = self.coord_loss(
                lhand_bbox_center,
                targets["lhand_bbox_center"],
                meta_info["lhand_bbox_valid"][:, None],
            ) + self.coord_loss(
                lhand_bbox_size,
                targets["lhand_bbox_size"],
                meta_info["lhand_bbox_valid"][:, None],
            )
            loss["rhand_bbox"] = self.coord_loss(
                rhand_bbox_center,
                targets["rhand_bbox_center"],
                meta_info["rhand_bbox_valid"][:, None],
            ) + self.coord_loss(
                rhand_bbox_size,
                targets["rhand_bbox_size"],
                meta_info["rhand_bbox_valid"][:, None],
            )
            # loss["face_bbox"] = self.coord_loss(
            #     face_bbox_center,
            #     targets["face_bbox_center"],
            #     meta_info["face_bbox_valid"][:, None],
            # ) + self.coord_loss(
            #     face_bbox_size,
            #     targets["face_bbox_size"],
            #     meta_info["face_bbox_valid"][:, None],
            # )

            # 改变手部目标关节图像和截断关节根据手部边界框（cfg.output_hm_shape -> 手部边界框空间）
            for part_name, bbox in (("lhand", lhand_bbox), ("rhand", rhand_bbox)):
                for coord_name, trunc_name in (
                    ("joint_img", "joint_trunc"),
                    ("smplx_joint_img", "smplx_joint_trunc"),
                ):
                    x = targets[coord_name][:, smpl_x.joint_part[part_name], 0]
                    y = targets[coord_name][:, smpl_x.joint_part[part_name], 1]
                    z = targets[coord_name][:, smpl_x.joint_part[part_name], 2]
                    trunc = meta_info[trunc_name][:, smpl_x.joint_part[part_name], 0]

                    x -= (
                        bbox[:, None, 0]
                        / cfg.input_body_shape[1]
                        * cfg.output_hm_shape[2]
                    )
                    x *= cfg.output_hand_hm_shape[2] / (
                        (bbox[:, None, 2] - bbox[:, None, 0])
                        / cfg.input_body_shape[1]
                        * cfg.output_hm_shape[2]
                    )
                    y -= (
                        bbox[:, None, 1]
                        / cfg.input_body_shape[0]
                        * cfg.output_hm_shape[1]
                    )
                    y *= cfg.output_hand_hm_shape[1] / (
                        (bbox[:, None, 3] - bbox[:, None, 1])
                        / cfg.input_body_shape[0]
                        * cfg.output_hm_shape[1]
                    )
                    z *= cfg.output_hand_hm_shape[0] / cfg.output_hm_shape[0]
                    trunc *= (
                        (x >= 0)
                        * (x < cfg.output_hand_hm_shape[2])
                        * (y >= 0)
                        * (y < cfg.output_hand_hm_shape[1])
                    )

                    coord = torch.stack((x, y, z), 2)
                    trunc = trunc[:, :, None]
                    targets[coord_name] = torch.cat(
                        (
                            targets[coord_name][
                                :, : smpl_x.joint_part[part_name][0], :
                            ],
                            coord,
                            targets[coord_name][
                                :, smpl_x.joint_part[part_name][-1] + 1 :, :
                            ],
                        ),
                        1,
                    )
                    meta_info[trunc_name] = torch.cat(
                        (
                            meta_info[trunc_name][
                                :, : smpl_x.joint_part[part_name][0], :
                            ],
                            trunc,
                            meta_info[trunc_name][
                                :, smpl_x.joint_part[part_name][-1] + 1 :, :
                            ],
                        ),
                        1,
                    )

            # 改变手部投影关节坐标根据手部边界框（cfg.output_hm_shape -> 手部边界框空间）
            for part_name, bbox in (("lhand", lhand_bbox), ("rhand", rhand_bbox)):
                x = joint_proj[:, smpl_x.joint_part[part_name], 0]
                y = joint_proj[:, smpl_x.joint_part[part_name], 1]

                x -= bbox[:, None, 0] / cfg.input_body_shape[1] * cfg.output_hm_shape[2]
                x *= cfg.output_hand_hm_shape[2] / (
                    (bbox[:, None, 2] - bbox[:, None, 0])
                    / cfg.input_body_shape[1]
                    * cfg.output_hm_shape[2]
                )
                y -= bbox[:, None, 1] / cfg.input_body_shape[0] * cfg.output_hm_shape[1]
                y *= cfg.output_hand_hm_shape[1] / (
                    (bbox[:, None, 3] - bbox[:, None, 1])
                    / cfg.input_body_shape[0]
                    * cfg.output_hm_shape[1]
                )

                coord = torch.stack((x, y), 2)
                trans = []
                for bid in range(coord.shape[0]):
                    mask = (
                        meta_info["joint_trunc"][bid, smpl_x.joint_part[part_name], 0]
                        == 1
                    )
                    if torch.sum(mask) == 0:
                        trans.append(torch.zeros((2)).float().cuda())
                    else:
                        trans.append(
                            (
                                -coord[bid, mask, :2]
                                + targets["joint_img"][
                                    :, smpl_x.joint_part[part_name], :
                                ][bid, mask, :2]
                            ).mean(0)
                        )
                trans = torch.stack(trans)[:, None, :]
                coord = coord + trans  # global translation alignment
                joint_proj = torch.cat(
                    (
                        joint_proj[:, : smpl_x.joint_part[part_name][0], :],
                        coord,
                        joint_proj[:, smpl_x.joint_part[part_name][-1] + 1 :, :],
                    ),
                    1,
                )

            # 改变面部投影关节坐标根据面部边界框（cfg.output_hm_shape -> 面部边界框空间）
            coord = joint_proj[:, smpl_x.joint_part["face"], :]
            trans = []
            for bid in range(coord.shape[0]):
                mask = meta_info["joint_trunc"][bid, smpl_x.joint_part["face"], 0] == 1
                if torch.sum(mask) == 0:
                    trans.append(torch.zeros((2)).float().cuda())
                else:
                    trans.append(
                        (
                            -coord[bid, mask, :2]
                            + targets["joint_img"][:, smpl_x.joint_part["face"], :][
                                bid, mask, :2
                            ]
                        ).mean(0)
                    )
            trans = torch.stack(trans)[:, None, :]
            coord = coord + trans  # global translation alignment
            joint_proj = torch.cat(
                (
                    joint_proj[:, : smpl_x.joint_part["face"][0], :],
                    coord,
                    joint_proj[:, smpl_x.joint_part["face"][-1] + 1 :, :],
                ),
                1,
            )

            loss["joint_proj"] = self.coord_loss(
                joint_proj, targets["joint_img"][:, :, :2], meta_info["joint_trunc"]
            )
            loss["joint_img"] = self.coord_loss(
                joint_img,
                smpl_x.reduce_joint_set(targets["joint_img"]),
                smpl_x.reduce_joint_set(meta_info["joint_trunc"]),
                meta_info["is_3D"],
            )
            loss["smplx_joint_img"] = self.coord_loss(
                joint_img,
                smpl_x.reduce_joint_set(targets["smplx_joint_img"]),
                smpl_x.reduce_joint_set(meta_info["smplx_joint_trunc"]),
            )
            return loss
        else:
            # 根据手部边界框更改手部输出 joint_img
            for part_name, bbox in (("lhand", lhand_bbox), ("rhand", rhand_bbox)):
                joint_img[:, smpl_x.pos_joint_part[part_name], 0] *= (
                    (bbox[:, None, 2] - bbox[:, None, 0])
                    / cfg.input_body_shape[1]
                    * cfg.output_hm_shape[2]
                ) / cfg.output_hand_hm_shape[2]
                joint_img[:, smpl_x.pos_joint_part[part_name], 0] += (
                    bbox[:, None, 0] / cfg.input_body_shape[1] * cfg.output_hm_shape[2]
                )
                joint_img[:, smpl_x.pos_joint_part[part_name], 1] *= (
                    (bbox[:, None, 3] - bbox[:, None, 1])
                    / cfg.input_body_shape[0]
                    * cfg.output_hm_shape[1]
                ) / cfg.output_hand_hm_shape[1]
                joint_img[:, smpl_x.pos_joint_part[part_name], 1] += (
                    bbox[:, None, 1] / cfg.input_body_shape[0] * cfg.output_hm_shape[1]
                )

            # 将手部边界框更改为输入图像空间
            for bbox in (lhand_bbox, rhand_bbox, face_bbox):
                bbox[:, 0] *= cfg.input_img_shape[1] / cfg.input_body_shape[1]
                bbox[:, 1] *= cfg.input_img_shape[0] / cfg.input_body_shape[0]
                bbox[:, 2] *= cfg.input_img_shape[1] / cfg.input_body_shape[1]
                bbox[:, 3] *= cfg.input_img_shape[0] / cfg.input_body_shape[0]

            out = {}  # 输出
            out["img"] = inputs["img"]
            out["joint_img"] = joint_img
            out["joint_cam"] = joint_cam
            out["smplx_joint_proj"] = joint_proj
            out["smplx_mesh_cam"] = mesh_cam
            out["smplx_root_pose"] = root_pose
            out["smplx_body_pose"] = body_pose
            out["smplx_lhand_pose"] = lhand_pose
            out["smplx_rhand_pose"] = rhand_pose
            out["smplx_jaw_pose"] = jaw_pose
            out["smplx_shape"] = shape
            out["smplx_expr"] = expr
            out["cam_trans"] = cam_trans
            out["lhand_bbox"] = lhand_bbox
            out["rhand_bbox"] = rhand_bbox
            out["face_bbox"] = face_bbox
            if "smplx_mesh_cam" in targets:
                out["smplx_mesh_cam_target"] = targets["smplx_mesh_cam"]
            if "smpl_mesh_cam" in targets:
                out["smpl_mesh_cam_target"] = targets["smpl_mesh_cam"]
            if "bb2img_trans" in meta_info:
                out["bb2img_trans"] = meta_info["bb2img_trans"]
            return out


def init_weights(m):
    """
    初始化神经网络中不同类型层的权重和偏差

    Args:
      m: 神经网络中的层
    """
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


def get_model(mode):
    """
    根据指定的模式（训练或测试）初始化并返回具有各种神经网络组件的模型

    Args:
      mode: 指定模型的模式，"train" 或 "test"
    Returns:
      使用指定的主干、位置网络、旋转网络、框网络、手部 ROI 网络和面部 ROI 网络创建的模型
    """
    backbone = ResNetBackbone(cfg.resnet_type)
    body_position_net = PositionNet("body", cfg.resnet_type)
    body_rotation_net = RotationNet("body", cfg.resnet_type)
    box_net = BoxNet()

    hand_backbone = ResNetBackbone(cfg.hand_resnet_type)
    hand_roi_net = HandRoI(hand_backbone)
    hand_position_net = PositionNet("hand", cfg.hand_resnet_type)
    hand_rotation_net = RotationNet("hand", cfg.hand_resnet_type)

    face_backbone = ResNetBackbone(cfg.face_resnet_type)
    face_roi_net = FaceRoI(face_backbone)
    face_regressor = FaceRegressor()

    if mode == "train":
        backbone.init_weights()
        body_position_net.apply(init_weights)
        body_rotation_net.apply(init_weights)
        box_net.apply(init_weights)

        hand_roi_net.apply(init_weights)
        hand_backbone.init_weights()
        hand_position_net.apply(init_weights)
        hand_rotation_net.apply(init_weights)

        face_roi_net.apply(init_weights)
        face_backbone.init_weights()
        face_regressor.apply(init_weights)

    model = Model(
        backbone,
        body_position_net,
        body_rotation_net,
        box_net,
        hand_roi_net,
        hand_position_net,
        hand_rotation_net,
        face_roi_net,
        face_regressor,
    )
    return model
