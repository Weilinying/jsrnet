import argparse
import os
import sys
import numpy as np
from tqdm import tqdm
import torch
from torchsummary import summary

from config import cfg
from datetime import datetime
import importlib

from mypath import Path
from dataloaders import make_data_loader
from net.sync_batchnorm.replicate import patch_replication_callback
from utils.calculate_weights import calculate_weigths_labels
from utils.lr_scheduler import LR_Scheduler
from utils.saver import Saver
from utils.summaries import TensorboardSummary
from utils.logger import Logger


class Trainer(object):
    def __init__(self, cfg):
        # 用配置设置初始化Trainer类
        # Define Saver
        self.saver = Saver(cfg)
        self.saver.save_experiment_config(cfg)  # 保存实验配置
        # Define Tensorboard Summary
        self.summary = TensorboardSummary(cfg, self.saver.experiment_dir)
        self.writer = self.summary.create_summary()
        #self.device = torch.device("cuda:{:d}".format(int(cfg.SYSTEM.GPU_IDS[0])) if cfg.SYSTEM.USE_GPU else "cpu")
        self.device = torch.device("cuda:0" if cfg.SYSTEM.USE_GPU else "cpu")  # 根据GPU可用性定义设备

        # Define Dataloader
        kwargs = {'num_workers': cfg.SYSTEM.NUM_CPU, 'pin_memory': True}
        self.train_loader, self.val_loader, self.test_loader, self.nclass = make_data_loader(cfg, **kwargs)

        # Define network
        kwargs = {'cfg': cfg, 'num_classes': self.nclass}  # 网络参数
        model_module = importlib.import_module("net.models")  # 动态导入模型模块
        self.model = getattr(model_module, cfg.MODEL.NET)(**kwargs)  # 实例化模型
        
        # Define Optimizer
        train_params = []
        params1x = self.model.get_1x_lr_params()  # 正常学习率的参数
        if params1x is not None: 
            train_params.append({'params': params1x , 'lr': cfg.OPTIMIZER.LR})
        params10x = self.model.get_10x_lr_params()  # 十倍学习率的参数
        if params10x is not None: 
            train_params.append({'params': params10x , 'lr': cfg.OPTIMIZER.LR*10})
        if len(train_params) == 0:  # 如果没有指定参数，则使用默认的SGD优化器
            print ("SGD: Training all parameters of model with LR: {:0.5f}".format(cfg.OPTIMIZER.LR))
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=cfg.OPTIMIZER.LR, momentum=cfg.OPTIMIZER.MOMENTUM,
                                    weight_decay=cfg.OPTIMIZER.WEIGHT_DECAY, 
                                    nesterov=cfg.OPTIMIZER.NESTEROV)
        else:  # 否则，使用指定参数组的SGD优化器
            print ("SGD: Training selected parameters of model with LR: {}".format([d["lr"] for d in train_params]))
            self.optimizer = torch.optim.SGD(train_params, momentum=cfg.OPTIMIZER.MOMENTUM,
                                    weight_decay=cfg.OPTIMIZER.WEIGHT_DECAY, 
                                    nesterov=cfg.OPTIMIZER.NESTEROV)

        # Define Criterion + whether to use class balanced weights 根据是否使用类平衡权重定义标准（损失函数）
        if cfg.EXPERIMENT.USE_BALANCED_WEIGHTS:
            classes_weights_path = os.path.join(Path.dataset_root_dir(cfg.DATASET.TRAIN), cfg.DATASET.TRAIN + '_classes_weights.npy')  # 获取类权重文件路径
            if os.path.isfile(classes_weights_path):  # 如果文件存在，则加载类权重
                weight = np.load(classes_weights_path)
            else:  # 否则，计算类权重
                weight = calculate_weigths_labels(cfg.DATASET.TRAIN, self.train_loader, self.nclass)
            weight = torch.from_numpy(weight.astype(np.float32))  # 将权重转换为torch张量
        else:  # 如果不使用平衡权重，则权重为None
            weight = None
        kwargs = {'cfg': cfg, 'loss_cfg': cfg.LOSS, 'weight': weight, 'use_cuda': cfg.SYSTEM.USE_GPU}  # 定义损失函数参数
        criterion_module = importlib.import_module("net.loss")  # 动态导入损失模块
        self.criterion = getattr(criterion_module, cfg.LOSS.TYPE)(**kwargs)  # 实例化损失函数
       
        # Define Evaluator
        evaluation_module = importlib.import_module("utils.metrics")  # 动态导入评估模块
        kwargs = {"num_class": self.nclass}  # 定义评估器参数
        self.evaluator = getattr(evaluation_module, cfg.EXPERIMENT.EVAL_METRIC)(cfg, **kwargs)  # 实例化评估器

        # Define lr scheduler
        self.scheduler = LR_Scheduler(cfg.OPTIMIZER.LR_SCHEDULER, cfg.OPTIMIZER.LR,  # 实例化学习率调度器
                                      cfg.EXPERIMENT.EPOCHS, len(self.train_loader))

        # Using cuda
        # 如果使用CUDA并且有多个GPU，则使用DataParallel来并行处理
        if cfg.SYSTEM.USE_GPU and torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=cfg.SYSTEM.GPU_IDS)
            patch_replication_callback(self.model)

        self.start_epoch = 0  # 初始化起始epoch
        self.epochs = cfg.EXPERIMENT.EPOCHS  # 设置训练的总epoch数
        # Resuming checkpoint  恢复检查点
        self.best_pred = 0.0  # 初始化最佳预测值
        if cfg.EXPERIMENT.RESUME_CHECKPOINT is not None:
            if not os.path.isfile(cfg.EXPERIMENT.RESUME_CHECKPOINT):
                raise RuntimeError("=> no checkpoint found at '{}'" .format(cfg.EXPERIMENT.RESUME_CHECKPOINT))
            checkpoint = torch.load(cfg.EXPERIMENT.RESUME_CHECKPOINT, map_location="cpu")  # 加载检查点
            if cfg.SYSTEM.USE_GPU and torch.cuda.device_count() > 1:
                self.model.module.load_state_dict(checkpoint['state_dict'])  # 如果使用多GPU，则加载模型状态
            else:
                self.model.load_state_dict(checkpoint['state_dict'])  # 否则，直接加载模型状态
            if not cfg.DATASET.FT:  # 如果不进行微调
                self.start_epoch = checkpoint['epoch']  # 更新起始epoch
                self.optimizer.load_state_dict(checkpoint['optimizer'])  # 加载优化器状态
                if cfg.SYSTEM.USE_GPU:
                    for state in self.optimizer.state.values():
                        for k, v in state.items():
                            if torch.is_tensor(v):
                                state[k] = v.to(self.device)  # 将优化器状态移动到指定设备
            self.best_pred = checkpoint['best_pred']  # 更新最佳预测值
            print("=> loaded checkpoint '{}' (epoch {})".format(cfg.EXPERIMENT.RESUME_CHECKPOINT, checkpoint['epoch']))
        self.model.to(self.device)  # 将模型移动到指定设备

    # 定义训练过程
    def training(self, epoch):
        train_loss = 0.0  # 初始化训练损失
        self.model.train()  # 设置模型为训练模式
        tbar = tqdm(self.train_loader)  # 进度条
        num_img_tr = len(self.train_loader)  # 获取训练数据的总数
        for i, sample in enumerate(tbar):  # 遍历训练数据
            image, target = sample['image'], sample['label']  # 获取图像和标签
            if cfg.SYSTEM.USE_GPU:
                image, target = image.to(self.device), target.to(self.device)  # 将数据移动到指定的设备
            self.scheduler(self.optimizer, i, epoch, self.best_pred)  # 更新学习率
            self.optimizer.zero_grad()  # 清空梯度
            output = self.model(image)  # 前向传播
            loss = self.criterion(output, target)  # 计算损失
            loss.backward()  # 反向传播
            self.optimizer.step()  # 更新模型参数
            train_loss += loss.item()  # 累加损失
            tbar.set_description('Train loss: %.3f' % (train_loss / (i + 1)))  # 更新进度条的描述
            self.writer.add_scalar('train/total_loss_iter', loss.item(), i + num_img_tr * epoch)  # 记录损失

            # Show 10 * 3 inference results each epoch  每个epoch显示10*3次推理结果
            if i % (num_img_tr // 10) == 0 and epoch % 10 == 0:
                global_step = i + num_img_tr * epoch
                self.summary.visualize_image(self.writer, cfg.DATASET.TRAIN, image, target, output, global_step, epoch, i)

        self.writer.add_scalar('train/total_loss_epoch', train_loss, epoch)  # 记录每个epoch的总损失
        print('[Epoch: %d, numImages: %5d, Loss: %.3f]' % (epoch, i * cfg.INPUT.BATCH_SIZE_TRAIN + image.data.shape[0], train_loss))

    # 定义验证过程
    def validation(self, epoch):
        self.model.eval()  # 设置模型为评估模式
        self.evaluator.reset()  # 重置评估器
        tbar = tqdm(self.val_loader, desc='\r')  # 创建进度条
        test_loss = 0.0  # 初始化验证损失
        num_img_val = len(self.val_loader)  # 获取验证数据的总数
        with torch.no_grad():
            for i, sample in enumerate(tbar):  # 遍历验证数据
                image, target = sample['image'], sample['label']  # 获取图像和标签
                if cfg.SYSTEM.USE_GPU:
                    image, target = image.to(self.device), target.to(self.device)  # 将数据移动到指定的设备
                with torch.no_grad():  # 禁止计算梯度
                    output = self.model(image)  # 前向传播
                loss = self.criterion(output, target)  # 计算损失
                test_loss += loss.item()  # 累加损失
                tbar.set_description('Val loss: %.3f' % (test_loss / (i + 1)))  # 更新进度条的描述
                # Add batch sample into evaluator  将批次结果添加到评估器
                self.evaluator.add_batch(target, output)

                # Show 10 * 3 inference results each epoch
                if i % 5 == 0 and epoch % 10 == 0:
                    global_step = i + num_img_val * epoch
                    self.summary.visualize_image(self.writer, cfg.DATASET.TRAIN, image, target, output, global_step, epoch, i, validation=True)

        # Fast test during the training  快速测试训练过程中的性能
        self.writer.add_scalar('val/total_loss_epoch', test_loss, epoch)  # 记录每个epoch的总损失
        print('Validation:')
        print('[Epoch: %d, numImages: %5d, Loss: %.3f]' % (epoch, i * cfg.INPUT.BATCH_SIZE_TRAIN + image.data.shape[0], test_loss))
        new_pred = self.evaluator.compute_stats(writer=self.writer, epoch=epoch)  # 计算并记录评估指标

        if new_pred > self.best_pred:  # 如果当前性能超过之前的最佳性能
            is_best = True
            self.best_pred = new_pred  # 更新最佳预测值
        else:
            is_best = False 

        # 根据是否为最佳模型保存检查点
        self.saver.save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': self.model.module.state_dict() if torch.cuda.device_count() > 1 else self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'best_pred': self.best_pred,
        }, is_best)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="PyTorch Training")
    parser.add_argument('--exp_cfg', type=str, default=None, help='Configuration file for experiment (it overrides the default settings).')  # 添加实验配置文件的命令行参数
    parser.add_argument('--gpu_ids', type=str, nargs='*', default=None, help='ids of gpus to used for training')  # 添加用于训练的GPU ID的命令行参数
    args = parser.parse_args()  # 解析命令行参数

    # print(f"Config file path: {args.exp_cfg}")
    # with open(args.exp_cfg, 'r') as cfg_file:
    #     print(cfg_file.read())

    if args.exp_cfg is not None and os.path.isfile(args.exp_cfg):  # 如果提供了实验配置文件，则合并配置
        cfg.merge_from_file(args.exp_cfg)

    if args.gpu_ids is not None:
        cfg.SYSTEM.GPU_IDS = [int(i) for i in args.gpu_ids]  # 设置要使用的GPU ID

    if not torch.cuda.is_available(): 
        print ("GPU is disabled")  # 如果没有检测到GPU，则输出提示信息
        cfg.SYSTEM.USE_GPU = False  # 禁用GPU

    if cfg.MODEL.SYNC_BN is None:
        if cfg.SYSTEM.USE_GPU and len(cfg.SYSTEM.GPU_IDS) > 1:  # 如果使用多GPU，则启用同步批量归一化
            cfg.MODEL.SYNC_BN = True
        else:
            cfg.MODEL.SYNC_BN = False

    if cfg.INPUT.BATCH_SIZE_TRAIN is None:
        cfg.INPUT.BATCH_SIZE_TRAIN = 4 * len(cfg.SYSTEM.GPU_IDS)  # 根据GPU数量设置训练批次大小

    if cfg.INPUT.BATCH_SIZE_TEST is None:
        cfg.INPUT.BATCH_SIZE_TEST = cfg.INPUT.BATCH_SIZE_TRAIN  # 设置测试批次大小

    if cfg.EXPERIMENT.NAME is None:
        cfg.EXPERIMENT.NAME = datetime.now().strftime(r'%Y%m%d_%H%M%S.%f').replace('.','_')  # 根据当前时间生成实验名称

    sys.stdout = Logger(os.path.join(cfg.EXPERIMENT.OUT_DIR, cfg.EXPERIMENT.NAME))  # 设置日志输出
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(i) for i in cfg.SYSTEM.GPU_IDS])  # 设置CUDA可见设备
    
    cfg.freeze()  # 冻结配置，防止后续修改
    print(cfg)  # 输出当前配置信息

    # fix rng seeds  固定随机数种子，以确保实验的可复现性
    torch.manual_seed(cfg.SYSTEM.RNG_SEED)
    np.random.seed(cfg.SYSTEM.RNG_SEED)

    trainer = Trainer(cfg)

    print (trainer.model)  # 打印模型结构
    #summary(trainer.model, input_size=(3, cfg.INPUT.CROP_SIZE, cfg.INPUT.CROP_SIZE))

    print("Saving experiment to:", trainer.saver.experiment_dir)  # 输出实验保存路径
    print('Starting Epoch:', trainer.start_epoch)  # 输出开始的epoch
    print('Total Epoches:', trainer.epochs)  # 输出总epoch数
    for epoch in range(trainer.start_epoch, trainer.epochs):
        trainer.training(epoch)  # 训练
        if (epoch % cfg.EXPERIMENT.EVAL_INTERVAL) == (cfg.EXPERIMENT.EVAL_INTERVAL - 1):
            trainer.validation(epoch)  # 验证
    trainer.writer.close()  # 关闭tensorboard写入器

