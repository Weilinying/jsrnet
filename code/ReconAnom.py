import os
import sys
import torch
import importlib

#from config import get_cfg_defaults
sys.path.append("/root/autodl-tmp/jsrnet/code/config")  # 将配置文件目录添加到Python的搜索路径中，使得可以导入配置文件
from config.defaults import get_cfg_defaults

# 异常检测框架，能够加载特定的配置和模型检查点，然后评估给定图像的异常分数

class MethodEvaluator():
    def __init__(self, **kwargs):
        """ Model initialization. """
        raise NotImplementedError

    def evaluate(self, image):
        """ Implement forward pass for a particular method. Return anomaly score per pixel. """
        raise NotImplementedError


class ReconAnom(MethodEvaluator):
    def __init__(self, **kwargs) -> None:
        self.exp_dir = kwargs["exp_dir"]  # 从关键字参数中获取实验目录路径
        self.code_dir = os.path.join(self.exp_dir, "code")  # 构造代码目录的路径

        cfg_local = get_cfg_defaults()  # 获取默认配置

        # 检查并加载实验目录中的parameters.yaml配置文件
        if os.path.isfile(os.path.join(self.exp_dir, "parameters.yaml")):
            with open(os.path.join(self.exp_dir, "parameters.yaml"), 'r') as f:
                cc = cfg_local._load_cfg_from_yaml_str(f)  # 加载YAML配置文件
            cfg_local.merge_from_file(os.path.join(self.exp_dir, "parameters.yaml"))  # 合并配置
            cfg_local.EXPERIMENT.NAME = cc.EXPERIMENT.NAME  # 设置实验名称
        else:
            assert False, "Experiment directory does not contain parameters.yaml: {}".format(self.exp_dir)

        # 检查并设置模型检查点路径
        if (os.path.isfile(os.path.join(self.exp_dir, "checkpoints", "checkpoint-best.pth")) 
                and cfg_local.EXPERIMENT.RESUME_CHECKPOINT is None):
            cfg_local.EXPERIMENT.RESUME_CHECKPOINT = os.path.join(self.exp_dir, "checkpoints", "checkpoint-best.pth")
        elif (cfg_local.EXPERIMENT.RESUME_CHECKPOINT is None 
                or not os.path.isfile(cfg_local.EXPERIMENT.RESUME_CHECKPOINT)):
            assert False, "Experiment dir does not contain best checkpoint, or no checkpoint specified or specified checkpoint does not exist: {}".format(cfg_local.EXPERIMENT.RESUME_CHECKPOINT)

        # 检查GPU可用性并设置
        if not torch.cuda.is_available(): 
            print ("GPU is disabled")
            cfg_local.SYSTEM.USE_GPU = False

        # 设置是否使用同步批量归一化
        if cfg_local.MODEL.SYNC_BN is None:
            if cfg_local.SYSTEM.USE_GPU and len(cfg_local.SYSTEM.GPU_IDS) > 1:
                cfg_local.MODEL.SYNC_BN = True
            else:
                cfg_local.MODEL.SYNC_BN = False

        # 设置训练和测试的批次大小
        if cfg_local.INPUT.BATCH_SIZE_TRAIN is None:
            cfg_local.INPUT.BATCH_SIZE_TRAIN = 4 * len(cfg_local.SYSTEM.GPU_IDS)
        if cfg_local.INPUT.BATCH_SIZE_TEST is None:
            cfg_local.INPUT.BATCH_SIZE_TEST = cfg_local.INPUT.BATCH_SIZE_TRAIN

        # 冻结配置，防止进一步修改
        cfg_local.freeze()
        # 设置设备
        self.device = torch.device("cuda:0" if cfg_local.SYSTEM.USE_GPU else "cpu")

        self.mean_tensor = torch.FloatTensor(cfg_local.INPUT.NORM_MEAN)[None, :, None, None].to(self.device)  # 设置归一化的均值
        self.std_tensor = torch.FloatTensor(cfg_local.INPUT.NORM_STD)[None, :, None, None].to(self.device)  # 设置归一化的标准差

        # Define network  动态加载网络模型
        sys.path.insert(0, self.code_dir)  # 将代码目录加入到系统路径
        # 这行代码创建了一个字典kwargs，其中包含一个键'cfg'，对应的值是cfg_local。这个cfg_local是之前通过加载和合并配置文件得到的配置对象。这个字典将被用作后续初始化模型时传递给模型构造函数的参数。
        kwargs = {'cfg': cfg_local}
        # 创建一个模块加载规范
        spec = importlib.util.spec_from_file_location("models", os.path.join(self.exp_dir, "code", "net", "models.py"))
        model_module = spec.loader.load_module()  # 动态加载模型模块
        print (self.exp_dir, model_module)
        self.model = getattr(model_module, cfg_local.MODEL.NET)(**kwargs)  # 实例化模型
        sys.path = sys.path[1:]  # 删除通过sys.path.insert(0, self.code_dir)添加的最近的路径, 恢复了sys.path到导入models模块之前的状态

        # 加载模型检查点
        if cfg_local.EXPERIMENT.RESUME_CHECKPOINT is not None:
            if not os.path.isfile(cfg_local.EXPERIMENT.RESUME_CHECKPOINT):
                raise RuntimeError("=> no checkpoint found at '{}'" .format(cfg_local.EXPERIMENT.RESUME_CHECKPOINT))
            checkpoint = torch.load(cfg_local.EXPERIMENT.RESUME_CHECKPOINT, map_location="cpu")
            if cfg_local.SYSTEM.USE_GPU and torch.cuda.device_count() > 1:
                self.model.module.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})".format(cfg_local.EXPERIMENT.RESUME_CHECKPOINT, checkpoint['epoch']))
            del checkpoint
        else:
            raise RuntimeError("=> model checkpoint has to be provided for testing!")
        
        # Using cuda
        self.model.to(self.device)  # 将模型移动到指定的设备
        self.model.eval()  # 设置模型为评估模式

        # 清理可能由于动态加载而产生的模块
        to_del = []
        for k, v in sys.modules.items():
            if k[:3] == "net":
                to_del.append(k)
        for k in to_del:
            del sys.modules[k]

    def evaluate(self, image):
        # 对图像进行归一化处理，并通过模型获取异常分数
        img = (image.to(self.device) - self.mean_tensor)/self.std_tensor
        with torch.no_grad():
            output = self.model(img)
        return output["anomaly_score"][:, 0, ...]


if __name__ == "__main__":
    # 根据情况修改项目地址
    # params = {"exp_dir": "<PATH TO THE DIRECTORY CONTAINING 'code' and 'parameters.yaml'>"}
    params = {"exp_dir": "D:\\SoftwareEngineer\\JSRNet"}
    evaluator = ReconAnom(**params)
    img = torch.rand((2, 3, 1024, 1024)).cuda() 
    # img as tensor in shape [B, 3, H, W] with values in range (0,1)
    out = evaluator.evaluate(img)  # 评估图像并获取异常分数
    print (out)
