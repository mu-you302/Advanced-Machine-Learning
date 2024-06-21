import numpy as np
import cv2
import random
from config import cfg
import math
from utils.human_models import smpl_x, smpl
from utils.transforms import cam2pixel, transform_joint_to_other_db
from plyfile import PlyData, PlyElement
import torch


def load_img(path, order="RGB"):
    """
    从指定路径读取图像并将其转换为具有指定颜色顺序的 NumPy 数组

    Args:
      path: 字符串，表示要加载的图像的文件路径
      order: 指定图像数据的颜色通道顺序

    Returns:
      按照 `order` 参数指定的格式返回从指定路径加载的图像。如果 `order` 设置为“RGB”，则该函数以 RGB 格式返回图像。图像在返回前会转换为
        float32 数据类型的 numpy 数组
    """
    img = cv2.imread(path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if not isinstance(img, np.ndarray):
        raise IOError("Fail to read %s" % path)

    if order == "RGB":
        img = img[:, :, ::-1].copy()

    img = img.astype(np.float32)
    return img


def get_bbox(joint_img, joint_valid, extend_ratio=1.2):
    """
    根据输入的关节坐标和有效性掩码计算边界框，并可选择按指定的比例扩大该框

    Args:
      joint_img: numpy 数组，包含图像中关节的坐标。每行代表一个关节
      joint_valid: 布尔数组，表示哪些关节是有效的

    Returns:
      以 numpy 数组的形式返回一个边界框（bbox），其中包含左上角的坐标（xmin，ymin）以及边界框的宽度和高度（xmax - xmin，ymax - ymin）
    """

    x_img, y_img = joint_img[:, 0], joint_img[:, 1]
    x_img = x_img[joint_valid == 1]
    y_img = y_img[joint_valid == 1]
    xmin = min(x_img)
    ymin = min(y_img)
    xmax = max(x_img)
    ymax = max(y_img)

    x_center = (xmin + xmax) / 2.0
    width = xmax - xmin
    xmin = x_center - 0.5 * width * extend_ratio
    xmax = x_center + 0.5 * width * extend_ratio

    y_center = (ymin + ymax) / 2.0
    height = ymax - ymin
    ymin = y_center - 0.5 * height * extend_ratio
    ymax = y_center + 0.5 * height * extend_ratio

    bbox = np.array([xmin, ymin, xmax - xmin, ymax - ymin]).astype(np.float32)
    return bbox


def sanitize_bbox(bbox, img_width, img_height):
    """
    调整边界框坐标以适合图像边界
    """
    x, y, w, h = bbox
    x1 = np.max((0, x))
    y1 = np.max((0, y))
    x2 = np.min((img_width - 1, x1 + np.max((0, w - 1))))
    y2 = np.min((img_height - 1, y1 + np.max((0, h - 1))))
    if w * h > 0 and x2 > x1 and y2 > y1:
        bbox = np.array([x1, y1, x2 - x1, y2 - y1])
    else:
        bbox = None

    return bbox


def process_bbox(bbox, img_width, img_height):
    """
    以边界框、图像宽度和图像高度作为输入，清理边界框坐标，并调整边界框以保持纵横比，
    然后将修改后的边界框作为 float32 值的 numpy 数组返回
    """
    bbox = sanitize_bbox(bbox, img_width, img_height)
    if bbox is None:
        return bbox

    # aspect ratio preserving bbox
    w = bbox[2]
    h = bbox[3]
    c_x = bbox[0] + w / 2.0
    c_y = bbox[1] + h / 2.0
    aspect_ratio = cfg.input_img_shape[1] / cfg.input_img_shape[0]
    if w > aspect_ratio * h:
        h = w / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    bbox[2] = w * 1.25
    bbox[3] = h * 1.25
    bbox[0] = c_x - bbox[2] / 2.0
    bbox[1] = c_y - bbox[3] / 2.0

    bbox = bbox.astype(np.float32)
    return bbox


def get_aug_config():
    """
    使用指定的因子生成缩放、旋转、颜色调整和翻转的随机增强参数

    Returns:
      返回四个值：“scale”、“rot”、“color_scale”和“do_flip”
    """
    scale_factor = 0.25
    rot_factor = 30
    color_factor = 0.2

    scale = np.clip(np.random.randn(), -1.0, 1.0) * scale_factor + 1.0
    rot = (
        np.clip(np.random.randn(), -2.0, 2.0) * rot_factor
        if random.random() <= 0.6
        else 0
    )
    c_up = 1.0 + color_factor
    c_low = 1.0 - color_factor
    color_scale = np.array(
        [
            random.uniform(c_low, c_up),
            random.uniform(c_low, c_up),
            random.uniform(c_low, c_up),
        ]
    )
    do_flip = random.random() <= 0.5

    return scale, rot, color_scale, do_flip


def augmentation(img, bbox, data_split):
    """
    根据提供的数据分割将缩放、旋转、翻转和颜色缩放等数据增强技术应用于图像
    """
    if data_split == "train":
        scale, rot, color_scale, do_flip = get_aug_config()
    else:
        scale, rot, color_scale, do_flip = 1.0, 0.0, np.array([1, 1, 1]), False

    img, trans, inv_trans = generate_patch_image(
        img, bbox, scale, rot, do_flip, cfg.input_img_shape
    )
    img = np.clip(img * color_scale[None, None, :], 0, 255)
    return img, trans, inv_trans, rot, do_flip


def generate_patch_image(cvimg, bbox, scale, rot, do_flip, out_shape):
    """
    采用输入图像、边界框坐标、缩放、旋转、翻转标志和输出形状来生成变换后的图像补丁以及变换矩阵

    Args:
      cvimg: 表示图像的 NumPy 数组
      bbox: 表示图像中对象的边界框坐标
      scale: 指定生成补丁图像时应用于边界框尺寸的缩放因子
      rot: 表示图像块将旋转的旋转角度（以度为单位）
      do_flip: 布尔标志，用于确定在处理之前是否应水平翻转图像
      out_shape: 将从输入图像中提取的补丁图像的期望输出形状
    """
    img = cvimg.copy()
    img_height, img_width, img_channels = img.shape

    bb_c_x = float(bbox[0] + 0.5 * bbox[2])
    bb_c_y = float(bbox[1] + 0.5 * bbox[3])
    bb_width = float(bbox[2])
    bb_height = float(bbox[3])

    if do_flip:
        img = img[:, ::-1, :]
        bb_c_x = img_width - bb_c_x - 1

    trans = gen_trans_from_patch_cv(
        bb_c_x, bb_c_y, bb_width, bb_height, out_shape[1], out_shape[0], scale, rot
    )
    img_patch = cv2.warpAffine(
        img, trans, (int(out_shape[1]), int(out_shape[0])), flags=cv2.INTER_LINEAR
    )
    img_patch = img_patch.astype(np.float32)
    inv_trans = gen_trans_from_patch_cv(
        bb_c_x,
        bb_c_y,
        bb_width,
        bb_height,
        out_shape[1],
        out_shape[0],
        scale,
        rot,
        inv=True,
    )

    return img_patch, trans, inv_trans


def rotate_2d(pt_2d, rot_rad):
    """
    将二维点绕原点以指定的弧度旋转
    """
    x = pt_2d[0]
    y = pt_2d[1]
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    xx = x * cs - y * sn
    yy = x * sn + y * cs
    return np.array([xx, yy], dtype=np.float32)


def gen_trans_from_patch_cv(
    c_x, c_y, src_width, src_height, dst_width, dst_height, scale, rot, inv=False
):
    """
    根据指定的参数生成仿射变换矩阵，用于图像块对齐

    Args:
      c_x: 表示源图像块中心点的 x 坐标。
      c_y: 表示源图像中心点的 y 坐标
      src_width: 表示源图像块的宽度
      src_height: 表示您正在变换的源图像块的高度
      dst_width: 表示要将源图像或 patch 变换到的目标图像或 patch的宽度。
      dst_height: 表示要将源图像或补丁变换到的目标图像或补丁的高度
      scale: 表示源图像尺寸缩放因子
      rot: 表示将应用于转换的旋转角度（以度为单位）
      inv: 决定是否计算正向变换还是计算逆变换
    """
    # 按照scale缩放
    src_w = src_width * scale
    src_h = src_height * scale
    src_center = np.array([c_x, c_y], dtype=np.float32)

    # 旋转增强
    rot_rad = np.pi * rot / 180
    src_downdir = rotate_2d(np.array([0, src_h * 0.5], dtype=np.float32), rot_rad)
    src_rightdir = rotate_2d(np.array([src_w * 0.5, 0], dtype=np.float32), rot_rad)

    dst_w = dst_width
    dst_h = dst_height
    dst_center = np.array([dst_w * 0.5, dst_h * 0.5], dtype=np.float32)
    dst_downdir = np.array([0, dst_h * 0.5], dtype=np.float32)
    dst_rightdir = np.array([dst_w * 0.5, 0], dtype=np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = src_center
    src[1, :] = src_center + src_downdir
    src[2, :] = src_center + src_rightdir

    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = dst_center
    dst[1, :] = dst_center + dst_downdir
    dst[2, :] = dst_center + dst_rightdir

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    trans = trans.astype(np.float32)
    return trans


def process_db_coord(
    joint_img,
    joint_cam,
    joint_valid,
    do_flip,
    img_shape,
    flip_pairs,
    img2bb_trans,
    rot,
    src_joints_name,
    target_joints_name,
):
    """
    对关节坐标执行各种数据增强和转换

    Args:
      joint_img: 包含图像空间中关节二维坐标的数组
      joint_cam: numpy 数组，包含相机空间中关节的 3D 坐标
      joint_valid: 表示输入数据中每个关节的有效性
      do_flip: 是否应对数据应用翻转增强
      img_shape: 输入图像的形状
      flip_pairs: 翻转增强过程中需要交换的关节索引对列表
      img2bb_trans: 用于变换联合图像坐标的仿射变换矩阵
      rot: 3D 数据旋转增强步骤的旋转角度
      src_joints_name: 要转换到 `target_joints_name`指定的目标坐标系中的源坐标系的关节名称
      target_joints_name: 关节要转换到的数据库中的目标关节名称

    """
    joint_img, joint_cam, joint_valid = (
        joint_img.copy(),
        joint_cam.copy(),
        joint_valid.copy(),
    )

    # 翻转增强
    if do_flip:
        joint_cam[:, 0] = -joint_cam[:, 0]
        joint_img[:, 0] = img_shape[1] - 1 - joint_img[:, 0]
        for pair in flip_pairs:
            joint_img[pair[0], :], joint_img[pair[1], :] = (
                joint_img[pair[1], :].copy(),
                joint_img[pair[0], :].copy(),
            )
            joint_cam[pair[0], :], joint_cam[pair[1], :] = (
                joint_cam[pair[1], :].copy(),
                joint_cam[pair[0], :].copy(),
            )
            joint_valid[pair[0], :], joint_valid[pair[1], :] = (
                joint_valid[pair[1], :].copy(),
                joint_valid[pair[0], :].copy(),
            )

    # 三维旋转
    rot_aug_mat = np.array(
        [
            [np.cos(np.deg2rad(-rot)), -np.sin(np.deg2rad(-rot)), 0],
            [np.sin(np.deg2rad(-rot)), np.cos(np.deg2rad(-rot)), 0],
            [0, 0, 1],
        ],
        dtype=np.float32,
    )
    joint_cam = np.dot(rot_aug_mat, joint_cam.transpose(1, 0)).transpose(1, 0)

    # 仿射变换
    joint_img_xy1 = np.concatenate(
        (joint_img[:, :2], np.ones_like(joint_img[:, :1])), 1
    )
    joint_img[:, :2] = np.dot(img2bb_trans, joint_img_xy1.transpose(1, 0)).transpose(
        1, 0
    )
    joint_img[:, 0] = joint_img[:, 0] / cfg.input_img_shape[1] * cfg.output_hm_shape[2]
    joint_img[:, 1] = joint_img[:, 1] / cfg.input_img_shape[0] * cfg.output_hm_shape[1]

    # check truncation
    joint_trunc = joint_valid * (
        (joint_img[:, 0] >= 0)
        * (joint_img[:, 0] < cfg.output_hm_shape[2])
        * (joint_img[:, 1] >= 0)
        * (joint_img[:, 1] < cfg.output_hm_shape[1])
        * (joint_img[:, 2] >= 0)
        * (joint_img[:, 2] < cfg.output_hm_shape[0])
    ).reshape(-1, 1).astype(np.float32)

    # 转换关节名称
    joint_img = transform_joint_to_other_db(
        joint_img, src_joints_name, target_joints_name
    )
    joint_cam = transform_joint_to_other_db(
        joint_cam, src_joints_name, target_joints_name
    )
    joint_valid = transform_joint_to_other_db(
        joint_valid, src_joints_name, target_joints_name
    )
    joint_trunc = transform_joint_to_other_db(
        joint_trunc, src_joints_name, target_joints_name
    )
    return joint_img, joint_cam, joint_valid, joint_trunc


def process_human_model_output(
    human_model_param,
    cam_param,
    do_flip,
    img_shape,
    img2bb_trans,
    rot,
    human_model_type,
):
    """
    处理不同类型的人体模型的人体模型参数和相机参数，应用各种转换和数据增强
    Args:
      human_model_param:包含有关人体模型的信息
      cam_param: 包含相机参数, 用于将 3D 点投影到 2D 图像坐标中
      do_flip: 是否应应用涉及翻转的数据增强
      img_shape: 图像的形状
      img2bb_trans: 用于 x、y 仿射变换和根相对深度
      rot: 3D 数据旋转增强的旋转角度（以度为单位）
      human_model_type: 指定所使用的人体模型的类型
    """
    if human_model_type == "smplx":
        human_model = smpl_x
        rotation_valid = np.ones((smpl_x.orig_joint_num), dtype=np.float32)
        coord_valid = np.ones((smpl_x.joint_num), dtype=np.float32)

        root_pose, body_pose, shape, trans = (
            human_model_param["root_pose"],
            human_model_param["body_pose"],
            human_model_param["shape"],
            human_model_param["trans"],
        )
        if "lhand_pose" in human_model_param and human_model_param["lhand_valid"]:
            lhand_pose = human_model_param["lhand_pose"]
        else:
            lhand_pose = np.zeros(
                (3 * len(smpl_x.orig_joint_part["lhand"])), dtype=np.float32
            )
            rotation_valid[smpl_x.orig_joint_part["lhand"]] = 0
            coord_valid[smpl_x.joint_part["lhand"]] = 0
        if "rhand_pose" in human_model_param and human_model_param["rhand_valid"]:
            rhand_pose = human_model_param["rhand_pose"]
        else:
            rhand_pose = np.zeros(
                (3 * len(smpl_x.orig_joint_part["rhand"])), dtype=np.float32
            )
            rotation_valid[smpl_x.orig_joint_part["rhand"]] = 0
            coord_valid[smpl_x.joint_part["rhand"]] = 0
        if (
            "jaw_pose" in human_model_param
            and "expr" in human_model_param
            and human_model_param["face_valid"]
        ):
            jaw_pose = human_model_param["jaw_pose"]
            expr = human_model_param["expr"]
            expr_valid = True
        else:
            jaw_pose = np.zeros((3), dtype=np.float32)
            expr = np.zeros((smpl_x.expr_code_dim), dtype=np.float32)
            rotation_valid[smpl_x.orig_joint_part["face"]] = 0
            coord_valid[smpl_x.joint_part["face"]] = 0
            expr_valid = False
        if "gender" in human_model_param:
            gender = human_model_param["gender"]
        else:
            gender = "neutral"
        root_pose = torch.FloatTensor(root_pose).view(1, 3)  # (1,3)
        body_pose = torch.FloatTensor(body_pose).view(-1, 3)  # (21,3)
        lhand_pose = torch.FloatTensor(lhand_pose).view(-1, 3)  # (15,3)
        rhand_pose = torch.FloatTensor(rhand_pose).view(-1, 3)  # (15,3)
        jaw_pose = torch.FloatTensor(jaw_pose).view(-1, 3)  # (1,3)
        shape = torch.FloatTensor(shape).view(1, -1)  # SMPLX shape parameter
        expr = torch.FloatTensor(expr).view(1, -1)  # SMPLX expression parameter
        trans = torch.FloatTensor(trans).view(1, -1)  # translation vector

        # 应用相机外参（旋转），融合根姿势和相机旋转
        if "R" in cam_param:
            R = np.array(cam_param["R"], dtype=np.float32).reshape(3, 3)
            root_pose = root_pose.numpy()
            root_pose, _ = cv2.Rodrigues(root_pose)
            root_pose, _ = cv2.Rodrigues(np.dot(R, root_pose))
            root_pose = torch.from_numpy(root_pose).view(1, 3)

        # 生成 SMPLX 网格和关节坐标
        zero_pose = torch.zeros((1, 3)).float()  # eye poses
        with torch.no_grad():
            output = smpl_x.layer[gender](
                betas=shape,
                body_pose=body_pose.view(1, -1),
                global_orient=root_pose,
                transl=trans,
                left_hand_pose=lhand_pose.view(1, -1),
                right_hand_pose=rhand_pose.view(1, -1),
                jaw_pose=jaw_pose.view(1, -1),
                leye_pose=zero_pose,
                reye_pose=zero_pose,
                expression=expr,
            )
        mesh_cam = output.vertices[0].numpy()
        joint_cam = output.joints[0].numpy()[smpl_x.joint_idx, :]

        # 相机外参（平移）应用，补偿旋转（未取消原点到根关节的平移）
        if "R" in cam_param and "t" in cam_param:
            R, t = np.array(cam_param["R"], dtype=np.float32).reshape(3, 3), np.array(
                cam_param["t"], dtype=np.float32
            ).reshape(1, 3)
            root_cam = joint_cam[smpl_x.root_joint_idx, None, :]
            joint_cam = (
                joint_cam
                - root_cam
                + np.dot(R, root_cam.transpose(1, 0)).transpose(1, 0)
                + t
            )
            mesh_cam = (
                mesh_cam
                - root_cam
                + np.dot(R, root_cam.transpose(1, 0)).transpose(1, 0)
                + t
            )

        # 拼接根姿势、身体姿势、两只手和下巴姿势
        pose = torch.cat((root_pose, body_pose, lhand_pose, rhand_pose, jaw_pose))

        # 关节坐标
        joint_img = cam2pixel(joint_cam, cam_param["focal"], cam_param["princpt"])
        joint_cam = (
            joint_cam - joint_cam[smpl_x.root_joint_idx, None, :]
        )  # root-relative
        joint_cam[smpl_x.joint_part["lhand"], :] = (
            joint_cam[smpl_x.joint_part["lhand"], :]
            - joint_cam[smpl_x.lwrist_idx, None, :]
        )  # left hand root-relative
        joint_cam[smpl_x.joint_part["rhand"], :] = (
            joint_cam[smpl_x.joint_part["rhand"], :]
            - joint_cam[smpl_x.rwrist_idx, None, :]
        )  # right hand root-relative
        joint_cam[smpl_x.joint_part["face"], :] = (
            joint_cam[smpl_x.joint_part["face"], :]
            - joint_cam[smpl_x.neck_idx, None, :]
        )  # 脸部相对根关节
        joint_img[smpl_x.joint_part["body"], 2] = (
            (
                joint_cam[smpl_x.joint_part["body"], 2].copy() / (cfg.body_3d_size / 2)
                + 1
            )
            / 2.0
            * cfg.output_hm_shape[0]
        )  # 身体深度离散化
        joint_img[smpl_x.joint_part["lhand"], 2] = (
            (
                joint_cam[smpl_x.joint_part["lhand"], 2].copy() / (cfg.hand_3d_size / 2)
                + 1
            )
            / 2.0
            * cfg.output_hm_shape[0]
        )  # 左手深度离散化
        joint_img[smpl_x.joint_part["rhand"], 2] = (
            (
                joint_cam[smpl_x.joint_part["rhand"], 2].copy() / (cfg.hand_3d_size / 2)
                + 1
            )
            / 2.0
            * cfg.output_hm_shape[0]
        )  # 右手深度离散化
        joint_img[smpl_x.joint_part["face"], 2] = (
            (
                joint_cam[smpl_x.joint_part["face"], 2].copy() / (cfg.face_3d_size / 2)
                + 1
            )
            / 2.0
            * cfg.output_hm_shape[0]
        )  # 人脸深度离散化

    elif human_model_type == "smpl":
        human_model = smpl
        pose, shape, trans = (
            human_model_param["pose"],
            human_model_param["shape"],
            human_model_param["trans"],
        )
        if "gender" in human_model_param:
            gender = human_model_param["gender"]
        else:
            gender = "neutral"
        pose = torch.FloatTensor(pose).view(-1, 3)
        shape = torch.FloatTensor(shape).view(1, -1)
        trans = torch.FloatTensor(trans).view(1, -1)  # 平移向量

        # 应用相机外参（旋转），融合根姿势和相机旋转
        if "R" in cam_param:
            R = np.array(cam_param["R"], dtype=np.float32).reshape(3, 3)
            root_pose = pose[smpl.orig_root_joint_idx, :].numpy()
            root_pose, _ = cv2.Rodrigues(root_pose)
            root_pose, _ = cv2.Rodrigues(np.dot(R, root_pose))
            pose[smpl.orig_root_joint_idx] = torch.from_numpy(root_pose).view(3)

        # 获取网格和关节坐标
        root_pose = pose[smpl.orig_root_joint_idx].view(1, 3)
        body_pose = torch.cat(
            (
                pose[: smpl.orig_root_joint_idx, :],
                pose[smpl.orig_root_joint_idx + 1 :, :],
            )
        ).view(1, -1)
        with torch.no_grad():
            output = smpl.layer[gender](
                betas=shape, body_pose=body_pose, global_orient=root_pose, transl=trans
            )
        mesh_cam = output.vertices[0].numpy()
        joint_cam = np.dot(smpl.joint_regressor, mesh_cam)

        # 应用相机外参（平移），补偿旋转（未取消原点到根关节的平移）
        if "R" in cam_param and "t" in cam_param:
            R, t = np.array(cam_param["R"], dtype=np.float32).reshape(3, 3), np.array(
                cam_param["t"], dtype=np.float32
            ).reshape(1, 3)
            root_cam = joint_cam[smpl.root_joint_idx, None, :]
            joint_cam = (
                joint_cam
                - root_cam
                + np.dot(R, root_cam.transpose(1, 0)).transpose(1, 0)
                + t
            )
            mesh_cam = (
                mesh_cam
                - root_cam
                + np.dot(R, root_cam.transpose(1, 0)).transpose(1, 0)
                + t
            )

        # 关节坐标
        joint_img = cam2pixel(joint_cam, cam_param["focal"], cam_param["princpt"])
        joint_cam = (
            joint_cam - joint_cam[smpl.root_joint_idx, None, :]
        )  # 身体相对根关节
        joint_img[:, 2] = (
            (joint_cam[:, 2].copy() / (cfg.body_3d_size / 2) + 1)
            / 2.0
            * cfg.output_hm_shape[0]
        )  # 身体深度离散化

    elif human_model_type == "mano":
        human_model = mano
        pose, shape, trans = (
            human_model_param["pose"],
            human_model_param["shape"],
            human_model_param["trans"],
        )
        hand_type = human_model_param["hand_type"]
        pose = torch.FloatTensor(pose).view(-1, 3)
        shape = torch.FloatTensor(shape).view(1, -1)
        trans = torch.FloatTensor(trans).view(1, -1)  # translation vector (1,3)

        # 应用相机外参（旋转），融合根姿势和相机旋转
        if "R" in cam_param:
            R = np.array(cam_param["R"], dtype=np.float32).reshape(3, 3)
            root_pose = pose[mano.orig_root_joint_idx, :].numpy()
            root_pose, _ = cv2.Rodrigues(root_pose)
            root_pose, _ = cv2.Rodrigues(np.dot(R, root_pose))
            pose[mano.orig_root_joint_idx] = torch.from_numpy(root_pose).view(3)

        # get mesh and joint coordinates
        root_pose = pose[mano.orig_root_joint_idx].view(1, 3)
        hand_pose = torch.cat(
            (
                pose[: mano.orig_root_joint_idx, :],
                pose[mano.orig_root_joint_idx + 1 :, :],
            )
        ).view(1, -1)
        with torch.no_grad():
            output = mano.layer[hand_type](
                betas=shape, hand_pose=hand_pose, global_orient=root_pose, transl=trans
            )
        mesh_cam = output.vertices[0].numpy()
        joint_cam = np.dot(mano.joint_regressor, mesh_cam)

        # 应用相机外参（平移），补偿旋转（未取消原点到根关节的平移）
        if "R" in cam_param and "t" in cam_param:
            R, t = np.array(cam_param["R"], dtype=np.float32).reshape(3, 3), np.array(
                cam_param["t"], dtype=np.float32
            ).reshape(1, 3)
            root_cam = joint_cam[mano.root_joint_idx, None, :]
            joint_cam = (
                joint_cam
                - root_cam
                + np.dot(R, root_cam.transpose(1, 0)).transpose(1, 0)
                + t
            )
            mesh_cam = (
                mesh_cam
                - root_cam
                + np.dot(R, root_cam.transpose(1, 0)).transpose(1, 0)
                + t
            )

        # 3D joint to 2D joint
        joint_img = cam2pixel(joint_cam, cam_param["focal"], cam_param["princpt"])
        joint_cam = (
            joint_cam - joint_cam[mano.root_joint_idx, None, :]
        )  # hand root-relative
        joint_img[:, 2] = (
            (joint_cam[:, 2].copy() / (cfg.hand_3d_size / 2) + 1)
            / 2.0
            * cfg.output_hm_shape[0]
        )  # hand depth discretize

    mesh_cam_orig = mesh_cam.copy()  # back-up the original one

    ## 进行数据增强和转换

    # 图像反转
    if do_flip:
        joint_cam[:, 0] = -joint_cam[:, 0]
        joint_img[:, 0] = img_shape[1] - 1 - joint_img[:, 0]
        for pair in human_model.flip_pairs:
            joint_cam[pair[0], :], joint_cam[pair[1], :] = (
                joint_cam[pair[1], :].copy(),
                joint_cam[pair[0], :].copy(),
            )
            joint_img[pair[0], :], joint_img[pair[1], :] = (
                joint_img[pair[1], :].copy(),
                joint_img[pair[0], :].copy(),
            )
            if human_model_type == "smplx":
                coord_valid[pair[0]], coord_valid[pair[1]] = (
                    coord_valid[pair[1]].copy(),
                    coord_valid[pair[0]].copy(),
                )

    # x,y 仿射变换，根关节相对深度
    joint_img_xy1 = np.concatenate(
        (joint_img[:, :2], np.ones_like(joint_img[:, 0:1])), 1
    )
    joint_img[:, :2] = np.dot(img2bb_trans, joint_img_xy1.transpose(1, 0)).transpose(
        1, 0
    )[:, :2]
    joint_img[:, 0] = joint_img[:, 0] / cfg.input_img_shape[1] * cfg.output_hm_shape[2]
    joint_img[:, 1] = joint_img[:, 1] / cfg.input_img_shape[0] * cfg.output_hm_shape[1]

    # check truncation
    joint_trunc = (
        (
            (joint_img[:, 0] >= 0)
            * (joint_img[:, 0] < cfg.output_hm_shape[2])
            * (joint_img[:, 1] >= 0)
            * (joint_img[:, 1] < cfg.output_hm_shape[1])
            * (joint_img[:, 2] >= 0)
            * (joint_img[:, 2] < cfg.output_hm_shape[0])
        )
        .reshape(-1, 1)
        .astype(np.float32)
    )
    # 三维数据旋转增强
    rot_aug_mat = np.array(
        [
            [np.cos(np.deg2rad(-rot)), -np.sin(np.deg2rad(-rot)), 0],
            [np.sin(np.deg2rad(-rot)), np.cos(np.deg2rad(-rot)), 0],
            [0, 0, 1],
        ],
        dtype=np.float32,
    )
    # 坐标系
    joint_cam = np.dot(rot_aug_mat, joint_cam.transpose(1, 0)).transpose(1, 0)
    # 轴角旋转参数翻转
    if do_flip:
        for pair in human_model.orig_flip_pairs:
            pose[pair[0], :], pose[pair[1], :] = (
                pose[pair[1], :].clone(),
                pose[pair[0], :].clone(),
            )
            if human_model_type == "smplx":
                rotation_valid[pair[0]], rotation_valid[pair[1]] = (
                    rotation_valid[pair[1]].copy(),
                    rotation_valid[pair[0]].copy(),
                )
        pose[:, 1:3] *= -1  # 翻转y、z轴

    # 旋转根关节姿势
    pose = pose.numpy()
    root_pose = pose[human_model.orig_root_joint_idx, :]
    root_pose, _ = cv2.Rodrigues(root_pose)
    root_pose, _ = cv2.Rodrigues(np.dot(rot_aug_mat, root_pose))
    pose[human_model.orig_root_joint_idx] = root_pose.reshape(3)

    # 转换为平均形状
    shape[(shape.abs() > 3).any(dim=1)] = 0.0
    shape = shape.numpy().reshape(-1)

    if human_model_type == "smplx":
        pose = pose.reshape(-1)
        expr = expr.numpy().reshape(-1)
        return (
            joint_img,
            joint_cam,
            joint_trunc,
            pose,
            shape,
            expr,
            rotation_valid,
            coord_valid,
            expr_valid,
            mesh_cam_orig,
        )
    elif human_model_type == "smpl":
        pose = pose.reshape(-1)
        return joint_img, joint_cam, joint_trunc, pose, shape, mesh_cam_orig
    elif human_model_type == "mano":
        pose = pose.reshape(-1)
        return joint_img, joint_cam, joint_trunc, pose, shape, mesh_cam_orig


def get_fitting_error_3D(db_joint, db_joint_from_fit, joint_valid):
    """
    计算进行平移对齐后两组三维关节坐标之间的拟合误差

    Args:
      db_joint: 包含 3D 空间中的关节坐标的 numpy 数组
      db_joint_from_fit: NumPy 数组，包含从 3D 空间中的拟合过程获得的关节坐标
      joint_valid布尔数组，表示哪些关节对于计算是有效的

    Returns:
      在应用一些转换和对齐后，返回基于输入关节数据“db_joint”和“db_joint_from_fit”计算出的拟合误差
    """
    # 过滤数组重塑
    db_joint = db_joint[np.tile(joint_valid, (1, 3)) == 1].reshape(-1, 3)
    db_joint_from_fit = db_joint_from_fit[np.tile(joint_valid, (1, 3)) == 1].reshape(
        -1, 3
    )

    db_joint_from_fit = (
        db_joint_from_fit
        - np.mean(db_joint_from_fit, 0)[None, :]
        + np.mean(db_joint, 0)[None, :]
    )  # 平移对齐
    error = np.sqrt(np.sum((db_joint - db_joint_from_fit) ** 2, 1)).mean()
    return error


def load_obj(file_name):
    """
    读取包含 OBJ 格式的顶点坐标的文件并将其作为 NumPy 数组返回

    Args:
      file_name: obj 格式的文件

    Returns:
      返回一个包含从文件读取的所有顶点坐标的 NumPy 数组
    """
    v = []
    obj_file = open(file_name)
    for line in obj_file:
        words = line.split(" ")
        if words[0] == "v":
            x, y, z = float(words[1]), float(words[2]), float(words[3])
            v.append(np.array([x, y, z]))
    return np.stack(v)


def load_ply(file_name):
    """
    读取 PLY 文件并将顶点坐标作为 NumPy 数组返回

    Args:
      file_name: PLY 文件

    Returns:
      返回包含 PLY 文件中顶点坐标 (x, y, z) 的 NumPy 数组
    """
    plydata = PlyData.read(file_name)
    x = plydata["vertex"]["x"]
    y = plydata["vertex"]["y"]
    z = plydata["vertex"]["z"]
    v = np.stack((x, y, z), 1)
    return v
