# Advanced-Machine-Learning



## 1. 介绍
本仓库为《高级机器学习》课程作业代码，实现了。。。


## 2. Python环境配置
使用anaconda进行环境配置：
```
conda create -n AML python=3.8
conda activate AML
```

- 本项目使用PyTorch 1.7.1 + CUDA 11.0
```
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 -f https://download.pytorch.org/whl/torch_stable.html
```
- 安装其他依赖项：
```
pip install -r requirements.txt
conda install -y -c conda-forge ffmpeg
```

## 3. 目录结构
### 根目录的组织结构
```
.
|-- dataset
|-- demo
|-- main  
|-- models
|-- tool
|-- output  
|-- common
|   |-- utils
|   |   |-- human_model_files
|   |   |   |-- smpl
|   |   |   |   |-- SMPL_NEUTRAL.pkl
|   |   |   |   |-- SMPL_MALE.pkl
|   |   |   |   |-- SMPL_FEMALE.pkl
|   |   |   |-- smplx
|   |   |   |   |-- MANO_SMPLX_vertex_ids.pkl
|   |   |   |   |-- SMPL-X__FLAME_vertex_ids.npy
|   |   |   |   |-- SMPLX_NEUTRAL.pkl
|   |   |   |   |-- SMPLX_to_J14.pkl
|   |   |   |   |-- SMPLX_NEUTRAL.npz
|   |   |   |   |-- SMPLX_MALE.npz
|   |   |   |   |-- SMPLX_FEMALE.npz
|   |   |   |-- mano
|   |   |   |   |-- MANO_LEFT.pkl
|   |   |   |   |-- MANO_RIGHT.pkl
```

- dataset 包含数据集中的图像和注释，可以是软链接
- demo 包含演示代码
- main 包含用于训练和测试网络的代码以及配置文件
- models 包含模型的预训练权重
- tool 包含数据预处理和模型融合的代码
- output 包含训练的模型、输出结果、可视化结果、训练日志等输出信息
- common 基础代码，包含网络结构，数据集加载，数据处理和转换。其中`human_model_files`用于存放人体模型文件，可从[人体模型文件](https://pan.baidu.com/s/1P9NKJtzGJAkr62E0FJZvIA?pwd=9z6v)下载并以如上结构组织。


### 数据集的组织形式
在 dataset 文件夹中，各数据集的存放形式如下：
```
.
|-- dataset  
|   |-- EHF
|   |   |-- data
|   |   |   |-- EHF.json
|   |-- Human36M  
|   |   |-- images  
|   |   |-- annotations  
|   |-- COCOWB
|   |   |-- images  
|   |   |   |-- train2017  
|   |   |   |-- val2017  
|   |   |-- annotations 
|   |-- PW3D
|   |   |-- data
|   |   |   |-- 3DPW_train.json
|   |   |   |-- 3DPW_validation.json
|   |   |   |-- 3DPW_test.json
|   |   |-- imageFiles
|   |-- FreiHAND
|   |   |-- data
|   |   |   |-- training
|   |   |   |-- evaluation
|   |   |   |-- freihand_train_coco.json
|   |   |   |-- freihand_train_data.json
|   |   |   |-- freihand_eval_coco.json
|   |   |   |-- freihand_eval_data.json
```
下面提供各数据集的百度网盘下载链接
- EHF 数据集 https://pan.baidu.com/s/1mtKfEhyFc39f8S9GIlthtw?pwd=2bva
- Human3.6M 数据集 https://pan.baidu.com/s/1-BApMYUSU8AupC9q6cRtqw?pwd=qa7i
- COCOWB 数据集 
  
  图像可从官网下载：https://cocodataset.org/#keypoints-2017
  https://pan.baidu.com/s/1Lsk5wQLYoEbtmvnBDSPvAQ?pwd=8tgm
- 3DPW 数据集 https://pan.baidu.com/s/16zH3PsvbkSUUZzsQ3ZYEPw?pwd=bfte
- FreiHand 数据集：https://pan.baidu.com/s/1tgJ8zBAufiamoDjC1LY-XA?pwd=cger

### 输出文件夹
```
.
|-- output  
|   |-- log  
|   |-- model_dump  
|   |-- result  
|   |-- vis  
```
- log 包含日志文件
- model_dump 包含每个epoch结束后保存的模型权重
- result 测试阶段的模型输出
- vis 可视化结果

## 4. 快速开始
1. 首先下载预训练的模型权重：https://pan.baidu.com/s/1DoMUSBseuBiSIc5ku_DsJg?pwd=cdwp ，放到`models`文件夹内

2. 准备好 `common/utils/human_model_files` 中所需的人体模型文件，目录组织和下载方式上文已介绍

3. 准备测试图片放入demo文件夹中，例如已有的 `test.png`

4. 运行demo.py:
```
python demo.py --img test.png --gpu 0,1
```

运行完成后，会在demo文件夹下生成一个输入图片同名文件夹，结果包含：
- output.obj(三维模型文件)
- render_cropped_img.jpg（对目标人物裁剪后的渲染结果）
- render_original_img.jpg（原始图像渲染结果）
- smplx_param.json（结果中的SMPLX模型参数）

## 5. Train

1. 训练整个网络：
```
python train.py --gpu 0,1 --lr 1e-4 --end_epoch 7 --train_batch_size 16
```
命令行参数可以指定GPU、学习率、训练轮数，批大小

在`main/config.py` 中，trainset_3d, trainset2d 指定了使用的3D，2D数据集，此时为Human3.6M，COCOWB

2. 手部网络的特定训练
   

首先在在`main/config.py` 中修改使用的数据集为：
```
trainset_3d = ["FreiHand"]
trainset_2d = []
```
然后在FreiHand上开始训练：
```
python train.py --gpu 0,1 --lr 1e-4 --end_epoch 10 --train_batch_size 16 --parts hand
```

3. 模型融合
   

切换到 `tool` 目录，运行：
```
python merge_hand_to_all.py
```

会自动将上面训练得到的两种模型进行融合，如果设置了与本文不同的epoch，需要自行修改其中的路径。

4. 最后微调
   

重新指定`main/config.py`中的数据集：
```
trainset_3d = ["Human36M"]
trainset_2d = ["COCOWB"]
```
开始微调：
```
python train.py --gpu 0,1 --lr 1e-5 --end_epoch 7 --train_batch_size 16 --continue
```


## 6. Test
在`main/config.py`中指定测试集 testset，可以选择 EHF，PW3D，Human36M 进行测试

```
python test.py --gpu 0,1 --test_batch_size 24 --pretrained_model ../models/ckpt_6.pth.tar
```

## 7. 结果

原始图片：

<img src="./assets/test5.png" alt="test5" style="zoom:33%;" />

重建结果：

<img src="./assets/render_original_img.png" alt="render_original_img" style="zoom:33%;" />

更多结果详见报告