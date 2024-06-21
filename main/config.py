import os
import os.path as osp
import sys
import numpy as np


# 配置设置，包括数据集、模型设置、输入/输出形状、训练和测试配置以及目录路径
class Config:

    ## 使用的数据集
    trainset_3d = ["Human36M"]
    trainset_2d = ["COCOWB"]  # 不包含3D注释
    testset = "EHF"
    # testset = "PW3D"
    # testset = "Human36M"

    ## 基础模型
    resnet_type = 50
    hand_resnet_type = 50
    face_resnet_type = 18

    ## 输入输出形状
    input_img_shape = (512, 384)
    input_body_shape = (256, 192)
    output_hm_shape = (8, 8, 6)
    input_hand_shape = (256, 256)
    output_hand_hm_shape = (8, 8, 8)
    input_face_shape = (192, 192)
    focal = (5000, 5000)  # 虚拟焦距
    princpt = (
        input_body_shape[1] / 2,
        input_body_shape[0] / 2,
    )  # 虚拟的主点位置
    body_3d_size = 2
    hand_3d_size = 0.3
    face_3d_size = 0.3
    camera_3d_size = 2.5

    ## 训练配置
    lr_dec_factor = 10
    lr_dec_epoch = [4, 6]  # [40, 60] #[4,6]
    end_epoch = 7
    train_batch_size = 24

    ## 测试配置
    test_batch_size = 64

    ## 其他配置
    num_thread = 16
    gpu_ids = "0"
    num_gpus = 1
    continue_train = False

    ## 设置数据集和输出目录
    cur_dir = osp.dirname(os.path.abspath(__file__))
    root_dir = osp.join(cur_dir, "..")
    data_dir = osp.join(root_dir, "dataset")
    output_dir = osp.join(root_dir, "output")
    model_dir = osp.join(output_dir, "model_dump")
    vis_dir = osp.join(output_dir, "vis")
    log_dir = osp.join(output_dir, "log")
    result_dir = osp.join(output_dir, "result")
    human_model_path = osp.join(root_dir, "common", "utils", "human_model_files")

    def set_args(self, gpu_ids, lr=1e-4, continue_train=False):
        self.gpu_ids = gpu_ids
        self.num_gpus = len(self.gpu_ids.split(","))
        self.lr = float(lr)
        self.continue_train = continue_train
        os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu_ids
        print(">>> Using GPU: {}".format(self.gpu_ids))


cfg = Config()

sys.path.insert(0, osp.join(cfg.root_dir, "common"))
from utils.dir import add_pypath, make_folder

add_pypath(osp.join(cfg.data_dir))
for i in range(len(cfg.trainset_3d)):
    add_pypath(osp.join(cfg.data_dir, cfg.trainset_3d[i]))
for i in range(len(cfg.trainset_2d)):
    add_pypath(osp.join(cfg.data_dir, cfg.trainset_2d[i]))
add_pypath(osp.join(cfg.data_dir, cfg.testset))
make_folder(cfg.model_dir)
make_folder(cfg.vis_dir)
make_folder(cfg.log_dir)
make_folder(cfg.result_dir)
