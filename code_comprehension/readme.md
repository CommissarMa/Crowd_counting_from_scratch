# 各文件夹的意思
+ datasets: 存放各个数据集的代码及配置
    + GCC模块
        + [GCC.py](#GCC.py): 定义GCC数据集加载器
        + [loading_data.py](#): 加载GCC数据集的方法
        + [prepare_GCC.py](#): 生成GCC密度图的matlab代码
        + [setting.py](#GCC.py): 加载GCC数据集的配置参数  
        处理好的数据集的文件结构：
            + datasets
                + ProcessedData
                    + GCC
                        + GCC
                            + scene_**_*
                                + pngs_544_960
                                + csv_den_maps_k15s4_544_960
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
+ [train.py](#train.py): 训练
+ trainer.py

## Gcc.py
<div id="GCC.py"></div>

## config.py
<div id="config.py"></div>  

以下所有参数都存放于cfg字典中：  
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
EXP_NAME: 实验名称
EXP_PATH: 存储路径  
VAL_DENSE_START：开始频繁验证的epoch  
VAL_FREQ：在VAL_DENSE_START之前，验证频率为VAL_FREQ  
VISIBLE_NUM_IMGS: 可视化的图像数量，类似SHHA这样不同分辨率的数据集只能设置为1   

## train.py
<div id="train.py"></div>

准备seed和gpu  
准备dataloader  
准备model  
开始训练

