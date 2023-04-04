import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from net.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from net.aspp import build_aspp
from net.decoder import build_decoder
from net.backbone import build_backbone
from net.reconstruction_decoder import ReconstructionDecoder, ReconstructionDecoderModified
from net.sync_batchnorm.replicate import patch_replication_callback
from utils.gen_mask import gen_disjoint_mask
from config import cfg

# 模型的核心，提供了语义分割功能
class DeepLab(nn.Module):
    def __init__(self, cfg, num_classes):
        super(DeepLab, self).__init__()

        # 设置输出步长output_stride，这个参数决定了网络中空洞卷积的扩张率
        output_stride = cfg.MODEL.OUT_STRIDE
        if cfg.MODEL.BACKBONE == 'drn':
            output_stride = 8

        # 选择使用同步批量归一化或标准批量归一化
        if cfg.MODEL.SYNC_BN == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        # 构建backbone, ASPP模块, 解码器
        self.backbone = build_backbone(cfg.MODEL.BACKBONE, output_stride, BatchNorm)
        self.aspp = build_aspp(cfg.MODEL.BACKBONE, output_stride, BatchNorm)
        self.decoder = build_decoder(num_classes, cfg.MODEL.BACKBONE, BatchNorm)

        # 决定是否冻结批量归一化层
        self.freeze_bn = cfg.MODEL.FREEZE_BN

    def forward(self, input):
        # 前向传播定义
        x, low_level_feat = self.backbone(input)  # 从backbone获取特征
        x = self.aspp(x)  # ASPP处理
        x = self.decoder(x, low_level_feat)  # 解码器处理
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)  # 上采样到输入大小
        return x

    def freeze_bn(self):
        # 冻结批量归一化层
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()

    def get_1x_lr_params(self):
        # 获取低学习率参数
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if self.freeze_bn:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
                else:
                    if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                            or isinstance(m[1], nn.BatchNorm2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p

    def get_10x_lr_params(self):
        # 获取高学习率参数
        modules = [self.aspp, self.decoder]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if self.freeze_bn:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
                else:
                    if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                            or isinstance(m[1], nn.BatchNorm2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p

# 为模型提供了加载预训练模型和重建解码器的基础
class DeepLabCommon(nn.Module):
    def __init__(self, cfg, **kwargs):
        super(DeepLabCommon, self).__init__()
        self.freeze_bn = cfg.MODEL.FREEZE_BN 
        
        # inicialize trained segmentation model  初始化训练过的分割模型
        self.deeplab = DeepLab(cfg, cfg.MODEL.RECONSTRUCTION.SEGM_MODEL_NCLASS)
        if not os.path.isfile(cfg.MODEL.RECONSTRUCTION.SEGM_MODEL):
            raise RuntimeError("=> pretrained segmentation model not found at '{}'" .format(cfg.MODEL.RECONSTRUCTION.SEGM_MODEL))
        checkpoint = torch.load(cfg.MODEL.RECONSTRUCTION.SEGM_MODEL, map_location="cpu")
        self.deeplab.load_state_dict(checkpoint['state_dict'])  # 加载预训练模型
        for parameter in self.deeplab.parameters():
            parameter.requires_grad = False  # 冻结分割模型参数
        del checkpoint
        
        # Reconstruction block 
        # 1) reconstruction decoder from segmentation model encoder features through small bottleneck
        # 初始化重建解码器
        if cfg.MODEL.NET == 'DeepLabReconFuseSimpleTrainModified':
            self.recon_dec = ReconstructionDecoderModified(cfg)
        else:
            self.recon_dec = ReconstructionDecoder(cfg)

    def forward(self, input):
        return None 

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()

    def get_10x_lr_params(self):
        return None        
    
    def get_1x_lr_params(self):
        return None

# 扩展了DeepLabCommon，实现了使用预训练的DeepLab模型进行图像分割和重建的具体逻辑。
class DeepLabRecon(DeepLabCommon):
    def __init__(self, cfg, **kwargs):
        super(DeepLabRecon, self).__init__(cfg, **kwargs)


    def forward(self, input):
        with torch.no_grad():
            encoder_feat, low_level_feat = self.deeplab.backbone(input)
            x = self.deeplab.aspp(encoder_feat)
            x = self.deeplab.decoder(x, low_level_feat)
            segmentation = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)
        recon, recon_loss, bl = self.recon_dec(input, encoder_feat, low_level_feat)

        # 返回包含输入、分割、重建图像、重建损失、重建瓶颈和异常分数的字典
        return {"input": input,
                "segmentation": segmentation,
                "recon_img": recon,
                "recon_loss": recon_loss,
                "recon_bottleneck": bl,
                "anomaly_score": recon_loss, 
                }

# 通过深度学习模型进行语义分割，并结合图像重建损失进行异常检测。
class DeepLabReconFuseSimple(DeepLabCommon):
    def __init__(self, cfg, **kwargs):
        super(DeepLabReconFuseSimple, self).__init__(cfg, **kwargs)
        # 继承DeepLabCommon类，使用配置文件初始化

        # 2) merging of multiclass segmentation ouput and road reconstruction loss  融合卷积层：用于合并多类分割输出和重建损失
        self.fuse_conv = nn.Sequential(
               nn.Conv2d(cfg.MODEL.RECONSTRUCTION.SEGM_MODEL_NCLASS, 8, kernel_size=3, stride=1, padding=1),
               nn.BatchNorm2d(8),
               nn.ReLU(inplace=True),
               nn.Conv2d(8, 2, kernel_size=1, stride=1, padding=0),
               nn.BatchNorm2d(2),
               nn.ReLU(inplace=True))

    def forward(self, input):
        # 定义前向传播过程
        # 不计算梯度，减少内存消耗，加速计算
        with torch.no_grad():
            # 从backbone获取特征
            encoder_feat, low_level_feat = self.deeplab.backbone(input)
            # ASPP处理
            x = self.deeplab.aspp(encoder_feat)
            # 解码器处理
            x = self.deeplab.decoder(x, low_level_feat)
            # 上采样到输入图像的大小
            segmentation = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)
        # 重建解码器处理
        recon, recon_loss, bl = self.recon_dec(input, encoder_feat, low_level_feat)
        # 融合卷积层处理
        x = self.fuse_conv(segmentation)

        # 计算每个像素的异常分数
        perpixel = F.softmax(x * torch.cat([recon_loss, 1-recon_loss], dim=1), dim=1)[:, 0:1, ...]

        return {"input": input,
                "segmentation": segmentation,
                "binary_segmentation": x,
                "recon_img": recon,
                "recon_loss": recon_loss,
                "recon_bottleneck": bl,
                "anomaly_score": perpixel, 
                }

# 与DeepLabReconFuseSimple类似，这个类也是用于通过深度学习模型进行语义分割，并结合图像重建损失进行异常检测。不同之处在于它在融合卷积层中考虑了重建损失作为一个额外的通道输入
class DeepLabReconFuseSimpleTrain(DeepLabCommon):
    def __init__(self, cfg, **kwargs):
        super(DeepLabReconFuseSimpleTrain, self).__init__(cfg, **kwargs)
        
        # 2) merging of multiclass segmentation ouput and road reconstruction loss
        self.fuse_conv = nn.Sequential(
               nn.Conv2d(cfg.MODEL.RECONSTRUCTION.SEGM_MODEL_NCLASS+1, 8, kernel_size=3, stride=1, padding=1),
               nn.BatchNorm2d(8),
               nn.ReLU(inplace=True),
               nn.Conv2d(8, 2, kernel_size=1, stride=1, padding=0),
               nn.BatchNorm2d(2),
               nn.ReLU(inplace=True))

    def forward(self, input):
        with torch.no_grad():
            encoder_feat, low_level_feat = self.deeplab.backbone(input)
            x = self.deeplab.aspp(encoder_feat)
            x = self.deeplab.decoder(x, low_level_feat)
            segmentation = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)
        recon, recon_loss, bl = self.recon_dec(input, encoder_feat, low_level_feat)
        x = self.fuse_conv(torch.cat([segmentation, recon_loss], dim=1))
        
        perpixel = F.softmax(x, dim=1)[:, 0:1, ...]

        return {"input": input,
                "segmentation": segmentation,
                "binary_segmentation": x,
                "recon_img": recon,
                "recon_loss": recon_loss,
                "recon_bottleneck": bl,
                "anomaly_score": perpixel, 
                }

class DeepLabReconFuseSimpleTrainModified(DeepLabCommon):
    def __init__(self, cfg, **kwargs):
        super(DeepLabReconFuseSimpleTrainModified, self).__init__(cfg, **kwargs)

        # 2) merging of multiclass segmentation ouput and road reconstruction loss
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(cfg.MODEL.RECONSTRUCTION.SEGM_MODEL_NCLASS + 1, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 2, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True))

    def forward(self, input):
        with torch.no_grad():
            # 正常图片，先得到语义分割的结果
            encoder_feat, low_level_feat = self.deeplab.backbone(input)
            x = self.deeplab.aspp(encoder_feat)
            x = self.deeplab.decoder(x, low_level_feat)
            segmentation = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)

        with torch.no_grad():
            # 生成多个尺度的掩码，模拟不同大小的异常
            k_value = [16, 32, 64, 128]
            Ms_generator = gen_disjoint_mask(k_value, 3, cfg.INPUT.CROP_SIZE, cfg.INPUT.CROP_SIZE)
            Ms = next(Ms_generator)
            device = torch.device("cuda:0" if cfg.SYSTEM.USE_GPU else "cpu")
            # 应用掩码到输入图像，生成模拟异常的输入
            inputs = [input * (torch.tensor(mask, dtype=torch.float32, requires_grad=False).to(device)) for mask in Ms]
            encoder_feats=[]
            low_level_feats=[]
            # 对每个模拟异常的输入进行语义分割
            for x in inputs:
                encoder_feat,low_level_feat= self.deeplab.backbone(x)
                encoder_feats.append(encoder_feat)
                low_level_feats.append(low_level_feat)

        recon, recon_loss = self.recon_dec(Ms, input, inputs, encoder_feats, low_level_feats)

        x = self.fuse_conv(torch.cat([segmentation, recon_loss], dim=1))

        perpixel = F.softmax(x, dim=1)[:, 0:1, ...]

        return {"input": input,
                "segmentation": segmentation,
                "binary_segmentation": x,
                "recon_img": recon,
                "recon_loss": recon_loss,
                "anomaly_score": perpixel,
                }
