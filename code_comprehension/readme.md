# 各文件夹的意思
+ datasets: 存放各个数据集的代码及配置
    + GCC模块
        + [GCC.py](#GCC.py): 定义GCC数据集加载器
        + [loading_data.py](#): 加载GCC数据集的方法
        + [prepare_GCC.py](#): 生成GCC密度图的matlab代码
        + [setting.py](#GCC.py): 加载GCC数据集的配置参数
    + QNRF模块
    + SHHA模块
    + SHHB模块
    + UCF50模块
    + WE模型
    + get_density_map_gaussian.m: 生成密度图的matlab代码
    + pre_data.py: 生成密度图的python代码
+ misc: 杂项
    + pytorch_ssim
    + cal_mean.py:
    + layer.py:
    + ssim_loss:
    + transforms.py:
    + utils.py:
+ models: 模型
    + CC.py
+ results_reports: 保存实验结果的地方
+ [config.py](#config.py): 当前实验的参数设置
+ test.py
+ train.py
+ trainer.py

## Gcc.py
<div id="GCC.py"></div>

## config.py
<div id="config.py"></div>  

SEED: 复现所用的随机种子  
DATASET: 使用GCC, SHHA, SHHB, UCF50, QNRF, WE中的一个数据集  
NET: 使用MCNN, AlexNet, VGG, VGG_DECODER, Res50, CSRNet, SANet中的一个网络结构  
PRE_GCC: 是否使用GCC预训练过的模型  
PRE_GCC_MODEL: GCC预训练过的模型的路径  
RESUME：是否继续训练  
RESUME_PATH: 模型的路径  
GPU_ID: 单gpu: [0], [1] ...; 多gpus: [0,1]  
LR: 学习率
LR_DECAY: 学习率衰减  
LR_DECAY_START: 开始衰减的epoch
NUM_EPOCH_LR_DECAY: 衰减的频率
MAX_EPOCH: epoch数
LAMBDA_1: 多任务学习的权重，在单任务学习的模型中不使用  
PRINT_FREQ: 打印的频率  



