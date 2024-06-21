import torch
import numpy as np
import scipy
from config import cfg
from torch.nn import functional as F
import torchgeometry as tgm


def cam2pixel(cam_coord, f, c):
    x = cam_coord[:, 0] / cam_coord[:, 2] * f[0] + c[0]
    y = cam_coord[:, 1] / cam_coord[:, 2] * f[1] + c[1]
    z = cam_coord[:, 2]
    return np.stack((x, y, z), 1)


def pixel2cam(pixel_coord, f, c):
    x = (pixel_coord[:, 0] - c[0]) / f[0] * pixel_coord[:, 2]
    y = (pixel_coord[:, 1] - c[1]) / f[1] * pixel_coord[:, 2]
    z = pixel_coord[:, 2]
    return np.stack((x, y, z), 1)


def world2cam(world_coord, R, t):
    cam_coord = np.dot(R, world_coord.transpose(1, 0)).transpose(1, 0) + t.reshape(1, 3)
    return cam_coord


def cam2world(cam_coord, R, t):
    world_coord = np.dot(
        np.linalg.inv(R), (cam_coord - t.reshape(1, 3)).transpose(1, 0)
    ).transpose(1, 0)
    return world_coord


def rigid_transform_3D(A, B):
    """
    计算两组 3D 点 A 和 B 之间的刚性变换（旋转和平移）

    Args:
      A: 一个形状为 (n, 3) 的 numpy 数组，表示 3D 点的集合
      B: 一个形状为 (n, 3) 的 numpy 数组，表示 3D 点的集合

    Returns:
        c: 缩放因子
        R: 一个 3x3 的旋转矩阵
        t: 一个 3x1 的平移向量
    """
    n, dim = A.shape
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    H = np.dot(np.transpose(A - centroid_A), B - centroid_B) / n
    U, s, V = np.linalg.svd(H)
    R = np.dot(np.transpose(V), np.transpose(U))
    if np.linalg.det(R) < 0:
        s[-1] = -s[-1]
        V[2] = -V[2]
        R = np.dot(np.transpose(V), np.transpose(U))

    varP = np.var(A, axis=0).sum()
    c = 1 / varP * np.sum(s)

    t = -np.dot(c * R, np.transpose(centroid_A)) + np.transpose(centroid_B)
    return c, R, t


def rigid_align(A, B):
    """
    对一组点 A 执行刚性变换，以将它们与另一组点 B 对齐
    """
    c, R, t = rigid_transform_3D(A, B)
    A2 = np.transpose(np.dot(c * R, np.transpose(A))) + t
    return A2


def transform_joint_to_other_db(src_joint, src_name, dst_name):
    """
    根据匹配的关节名称将关节数据从一种数据库格式转换为另一种数据库格式

    Args:
      src_joint: 包含源数据库中关节数据的数组
      src_name: 包含源数据库中关节名称的列表
      dst_name: 目标数据库中的关节名称

    Returns:
      返回一个新的关节数组，其中来自源关节数组的关节数据被转换以匹配目标关节名称的顺序
    """
    src_joint_num = len(src_name)
    dst_joint_num = len(dst_name)

    new_joint = np.zeros(((dst_joint_num,) + src_joint.shape[1:]), dtype=np.float32)
    for src_idx in range(len(src_name)):
        name = src_name[src_idx]
        if name in dst_name:
            dst_idx = dst_name.index(name)
            new_joint[dst_idx] = src_joint[src_idx]

    return new_joint


def rot6d_to_axis_angle(x):
    """
    将一批 6D 旋转矩阵转换为轴角表示

    Args:
      以张量 `x` 作为输入，并执行一些操作以将 6D 旋转表示转换为轴角表示

    Returns:
      返回一批 6D 旋转矩阵的轴角表示
    """
    batch_size = x.shape[0]

    x = x.view(-1, 3, 2)
    a1 = x[:, :, 0]
    a2 = x[:, :, 1]
    b1 = F.normalize(a1)
    b2 = F.normalize(a2 - torch.einsum("bi,bi->b", b1, a2).unsqueeze(-1) * b1)
    b3 = torch.cross(b1, b2)
    rot_mat = torch.stack((b1, b2, b3), dim=-1)  # 3x3 rotation matrix

    rot_mat = torch.cat(
        [rot_mat, torch.zeros((batch_size, 3, 1)).cuda().float()], 2
    )  # 3x4 rotation matrix
    axis_angle = tgm.rotation_matrix_to_angle_axis(rot_mat).reshape(-1, 3)  # axis-angle
    axis_angle[torch.isnan(axis_angle)] = 0.0
    return axis_angle


def sample_joint_features(img_feat, joint_xy):
    """
    根据图像特征和关节坐标，转换关节坐标，使用转换后的坐标对图像特征进行采样，并在返回结果之前重新排列输出维度

    Args:
      img_feat: 从图像中提取的特征，形状为 (batch_size, channel_dim, height, width)
      joint_xy: 图像中关节的坐标。其形状为 (batch_size, joint_num, 2)

    Returns:

    输出是一个张量，其维度为（batch_size、joint_num、channel_dim），表示图像中关节的特征
    """
    height, width = img_feat.shape[2:]
    x = joint_xy[:, :, 0] / (width - 1) * 2 - 1
    y = joint_xy[:, :, 1] / (height - 1) * 2 - 1
    grid = torch.stack((x, y), 2)[:, :, None, :]
    img_feat = F.grid_sample(img_feat, grid, align_corners=True)[
        :, :, :, 0
    ]  # batch_size, channel_dim, joint_num
    img_feat = img_feat.permute(
        0, 2, 1
    ).contiguous()  # batch_size, joint_num, channel_dim
    return img_feat


def soft_argmax_2d(heatmap2d):
    """
    从 2D 热图计算软 argmax 坐标

    Args:
      以 4 维张量 `heatmap2d` 作为输入，并执行软 argmax 操作以根据热图值计算坐标

    Returns:
      返回坐标输出“coord_out”，其中包含基于输入2D热图的softmax计算的累积x和y坐标
    """
    batch_size = heatmap2d.shape[0]
    height, width = heatmap2d.shape[2:]
    heatmap2d = heatmap2d.reshape((batch_size, -1, height * width))
    heatmap2d = F.softmax(heatmap2d, 2)
    heatmap2d = heatmap2d.reshape((batch_size, -1, height, width))

    accu_x = heatmap2d.sum(dim=(2))
    accu_y = heatmap2d.sum(dim=(3))

    accu_x = accu_x * torch.arange(width).float().cuda()[None, None, :]
    accu_y = accu_y * torch.arange(height).float().cuda()[None, None, :]

    accu_x = accu_x.sum(dim=2, keepdim=True)
    accu_y = accu_y.sum(dim=2, keepdim=True)

    coord_out = torch.cat((accu_x, accu_y), dim=2)
    return coord_out


def soft_argmax_3d(heatmap3d):
    """
    使用softmax和加权和运算根据3D热图计算3D坐标

    Args:
      以 3D 热图张量作为输入，并执行软 argmax 操作以根据热图值计算坐标

    Returns:
      返回坐标输出“coord_out”，其中包含基于输入 3D 热图的 softmax 值计算的累积 x、y 和 z 坐标
    """
    batch_size = heatmap3d.shape[0]
    depth, height, width = heatmap3d.shape[2:]
    heatmap3d = heatmap3d.reshape((batch_size, -1, depth * height * width))
    heatmap3d = F.softmax(heatmap3d, 2)
    heatmap3d = heatmap3d.reshape((batch_size, -1, depth, height, width))

    accu_x = heatmap3d.sum(dim=(2, 3))
    accu_y = heatmap3d.sum(dim=(2, 4))
    accu_z = heatmap3d.sum(dim=(3, 4))

    accu_x = accu_x * torch.arange(width).float().cuda()[None, None, :]
    accu_y = accu_y * torch.arange(height).float().cuda()[None, None, :]
    accu_z = accu_z * torch.arange(depth).float().cuda()[None, None, :]

    accu_x = accu_x.sum(dim=2, keepdim=True)
    accu_y = accu_y.sum(dim=2, keepdim=True)
    accu_z = accu_z.sum(dim=2, keepdim=True)

    coord_out = torch.cat((accu_x, accu_y, accu_z), dim=2)
    return coord_out


def restore_bbox(bbox_center, bbox_size, aspect_ratio, extension_ratio):
    """
    根据纵横比和延伸比对bbox进行调整，并返回修改后的边界框坐标

    Args:
      bbox_center: (x_center, y_center) 格式表示边界框的中心坐标
      bbox_size: 边界框的大小，格式为 (width, height)
      aspect_ratio: 保留边界框的长宽比。如果边界框的宽度大于纵横比乘以高度，则将边界框的宽度调整为纵横比乘以高度。如果边界框的宽度小于纵横比乘以高度，则将边界框的高度调整为宽度除以纵横比。
      extension_ratio: 在保留纵横比的情况下扩展边界框的宽度和高度

    Returns:
       xyxy（左上角和右下角坐标）格式返回边界框
    """
    bbox = bbox_center.view(-1, 1, 2) + torch.cat(
        (-bbox_size.view(-1, 1, 2) / 2.0, bbox_size.view(-1, 1, 2) / 2.0), 1
    )  # xyxy in (cfg.output_hm_shape[2], cfg.output_hm_shape[1]) space
    bbox[:, :, 0] = bbox[:, :, 0] / cfg.output_hm_shape[2] * cfg.input_body_shape[1]
    bbox[:, :, 1] = bbox[:, :, 1] / cfg.output_hm_shape[1] * cfg.input_body_shape[0]
    bbox = bbox.view(-1, 4)

    # xyxy -> xywh
    bbox[:, 2] = bbox[:, 2] - bbox[:, 0]
    bbox[:, 3] = bbox[:, 3] - bbox[:, 1]

    # 纵横比保持边界框
    w = bbox[:, 2]
    h = bbox[:, 3]
    c_x = bbox[:, 0] + w / 2.0
    c_y = bbox[:, 1] + h / 2.0

    mask1 = w > (aspect_ratio * h)
    mask2 = w < (aspect_ratio * h)
    h[mask1] = w[mask1] / aspect_ratio
    w[mask2] = h[mask2] * aspect_ratio

    bbox[:, 2] = w * extension_ratio
    bbox[:, 3] = h * extension_ratio
    bbox[:, 0] = c_x - bbox[:, 2] / 2.0
    bbox[:, 1] = c_y - bbox[:, 3] / 2.0

    # xywh -> xyxy
    bbox[:, 2] = bbox[:, 2] + bbox[:, 0]
    bbox[:, 3] = bbox[:, 3] + bbox[:, 1]
    return bbox
