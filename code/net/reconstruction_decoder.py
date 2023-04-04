import numpy as np
import torch
from collections import OrderedDict
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import torchvision.models as models
from math import exp
from net.aspp import ASPP
from net.sync_batchnorm import SynchronizedBatchNorm2d


class ReconstructionDecoder(nn.Module):
    def __init__(self, cfg, **kwargs):
        # 初始化解码器，定义各层的通道数和潜在维度(latent_dim)
        super(ReconstructionDecoder, self).__init__()
        self.layers_dim = [16, 32, 64, 128]
        self.latent_dim = cfg.MODEL.RECONSTRUCTION.LATENT_DIM

        # 定义用于输入图像标准化的均值和标准差张量
        self.mean_tensor = torch.FloatTensor(cfg.INPUT.NORM_MEAN)[None, :, None, None]
        self.std_tensor = torch.FloatTensor(cfg.INPUT.NORM_STD)[None, :, None, None]

        # 选择批量归一化层类型
        self.BatchNorm = SynchronizedBatchNorm2d if cfg.MODEL.SYNC_BN else nn.BatchNorm2d

        # 定义一个ASPP模块作为瓶颈层，用于提取和压缩特征
        self.bottleneck = ASPP(cfg.MODEL.BACKBONE, cfg.MODEL.OUT_STRIDE, self.BatchNorm, outplanes=self.latent_dim)

        # 定义上采样层
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        #self.up_size = lambda x, sz: F.interpolate(x, size=sz, mode='bilinear', align_corners=True, recompute_scale_factor=True)
        self.up_size = lambda x, sz: F.interpolate(x, size=sz, mode='bilinear', align_corners=True)

        # 如果配置中启用了跳跃连接，定义一个转换层来匹配跳跃连接的维度
        self.skip_dim = 0 
        if cfg.MODEL.RECONSTRUCTION.SKIP_CONN:
            self.skip_dim = cfg.MODEL.RECONSTRUCTION.SKIP_CONN_DIM 
            self.skip_conn = nn.Sequential(nn.Conv2d(256, self.skip_dim, kernel_size=1, stride=1, padding=0, bias=False),
                                           nn.BatchNorm2d(self.skip_dim),
                                           nn.ReLU())

        # 定义解码层和最终的卷积层，用于将特征映射回图像空间
        self.dec_layer0 = conv_block(self.latent_dim, self.layers_dim[3])
        self.dec_layer1 = conv_block(self.layers_dim[3], self.layers_dim[2])
        self.dec_layer2 = conv_block(self.layers_dim[2]+self.skip_dim, self.layers_dim[1])
        self.dec_layer3 = conv_block(self.layers_dim[1], self.layers_dim[0])
        self.final_layer = nn.Conv2d(self.layers_dim[0], out_channels=3, kernel_size=1, stride=1, padding=0)

        # 初始化SSIM损失计算层
        self.recon_loss = SSIMLoss(window_size=11, absval=True)

    # 前向传播定义，包括特征重建和损失计算
    def forward(self, img, encoder_feat, low_level_feat):
        # 在forward方法中，确保mean_tensor和std_tensor与输入img在同一设备上
        if img.is_cuda:
            self.mean_tensor = self.mean_tensor.to(img.device)
            self.std_tensor = self.std_tensor.to(img.device)
        # 通过瓶颈层处理编码器特征
        bl = self.bottleneck(encoder_feat)
        # decoder # 逐层上采样和处理
        d_l0 = self.dec_layer0(self.up2(bl))
        d_l1 = self.dec_layer1(self.up2(d_l0))
        # 如果启用了跳跃连接，将其与解码器特征融合
        if self.skip_dim > 0:
            skip = F.interpolate(self.skip_conn(low_level_feat), size=d_l1.size()[2:], mode='bilinear', align_corners=True)
            d_l1 = torch.cat([d_l1, skip], dim=1)
         # 继续上采样和处理，直到达到原图像大小
        d_l2 = self.dec_layer2(self.up2(d_l1))
        d_l3 = self.dec_layer3(self.up_size(d_l2, img.shape[2:]))
        
        #print ("encoded feat: ", encoder_feat.size())
        #print ("LOW: ", low_level_feat.size())
        #print ("BL: ", bl.size())
        #print ("d0: ", d_l0.size())
        #print ("d1: ", d_l1.size())
        #print ("d2: ", d_l2.size())
        #print ("d3: ", d_l3.size())

        # 如果在GPU上运行，确保标准化张量也在相同的设备上
        if img.is_cuda and self.mean_tensor.get_device() != img.get_device():
            self.mean_tensor = self.mean_tensor.cuda(img.get_device())
            self.std_tensor = self.std_tensor.cuda(img.get_device())

        # 通过最终层生成重建图像，并进行标准化逆操作
        recons = torch.clamp(self.final_layer(d_l3)*self.std_tensor+self.mean_tensor, 0, 1)
        # 计算重建图像和原图像之间的SSIM损失
        recons_loss = self.recon_loss(recons, torch.clamp(img*self.std_tensor+self.mean_tensor, 0, 1))[:, None, ...]

        # 返回重建图像，重建损失和瓶颈层特征
        return [recons, recons_loss, bl]


# 只返回重构图
class ReconstructionDecoderV2(nn.Module):
    def __init__(self, cfg, **kwargs):
        super(ReconstructionDecoderV2, self).__init__()
        self.layers_dim = [16, 32, 64, 128]
        self.latent_dim = cfg.MODEL.RECONSTRUCTION.LATENT_DIM

        self.mean_tensor = torch.FloatTensor(cfg.INPUT.NORM_MEAN)[None, :, None, None]
        self.std_tensor = torch.FloatTensor(cfg.INPUT.NORM_STD)[None, :, None, None]

        if cfg.MODEL.SYNC_BN == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.bottleneck = ASPP(cfg.MODEL.BACKBONE, cfg.MODEL.OUT_STRIDE, BatchNorm, outplanes=self.latent_dim)

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # self.up_size = lambda x, sz: F.interpolate(x, size=sz, mode='bilinear', align_corners=True, recompute_scale_factor=True)
        self.up_size = lambda x, sz: F.interpolate(x, size=sz, mode='bilinear', align_corners=True)

        self.skip_dim = 0
        if cfg.MODEL.RECONSTRUCTION.SKIP_CONN:
            self.skip_dim = cfg.MODEL.RECONSTRUCTION.SKIP_CONN_DIM
            self.skip_conn = nn.Sequential(
                nn.Conv2d(256, self.skip_dim, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(self.skip_dim),
                nn.ReLU())

        self.dec_layer0 = conv_block(self.latent_dim, self.layers_dim[3])
        self.dec_layer1 = conv_block(self.layers_dim[3], self.layers_dim[2])
        self.dec_layer2 = conv_block(self.layers_dim[2] + self.skip_dim, self.layers_dim[1])
        self.dec_layer3 = conv_block(self.layers_dim[1], self.layers_dim[0])
        self.final_layer = nn.Conv2d(self.layers_dim[0], out_channels=3, kernel_size=1, stride=1, padding=0)

        self.recon_loss = SSIMLoss(window_size=11, absval=True)

    def forward(self, img, encoder_feat, low_level_feat):
        bl = self.bottleneck(encoder_feat)
        # decoder
        d_l0 = self.dec_layer0(self.up2(bl))
        d_l1 = self.dec_layer1(self.up2(d_l0))
        if self.skip_dim > 0:
            skip = F.interpolate(self.skip_conn(low_level_feat), size=d_l1.size()[2:], mode='bilinear',
                                 align_corners=True)
            d_l1 = torch.cat([d_l1, skip], dim=1)
        d_l2 = self.dec_layer2(self.up2(d_l1))
        d_l3 = self.dec_layer3(self.up_size(d_l2, img.shape[2:]))

        # print ("encoded feat: ", encoder_feat.size())
        # print ("LOW: ", low_level_feat.size())
        # print ("BL: ", bl.size())
        # print ("d0: ", d_l0.size())
        # print ("d1: ", d_l1.size())
        # print ("d2: ", d_l2.size())
        # print ("d3: ", d_l3.size())
        if img.is_cuda and self.mean_tensor.get_device() != img.get_device():
            self.mean_tensor = self.mean_tensor.cuda(img.get_device())
            self.std_tensor = self.std_tensor.cuda(img.get_device())

        recons = torch.clamp(self.final_layer(d_l3) * self.std_tensor + self.mean_tensor, 0, 1)
        # recons_loss = self.recon_loss(recons, torch.clamp(img * self.std_tensor + self.mean_tensor, 0, 1))[:, None, ...]

        return recons


# 修改的重构模块，将分割模块对多张mask的图片下采样的特征输入，以及原图，
class ReconstructionDecoderModified(nn.Module):
    def __init__(self, cfg, **kwargs):
        super(ReconstructionDecoderModified, self).__init__()
        # 利用原代码中的重建模块，得到重建的图
        self.realRecon = ReconstructionDecoderV2(cfg)
        self.recon_loss = SSIMLoss(window_size=11, absval=True)
        self.cfg = cfg
        self.mean_tensor = torch.FloatTensor(cfg.INPUT.NORM_MEAN)[None, :, None, None]
        self.std_tensor = torch.FloatTensor(cfg.INPUT.NORM_STD)[None, :, None, None]

    def forward(self, Ms, orginal_img, imgs, encoder_feats, low_level_feats):
        # Ms: 一组掩码，用于指定每个重建图像应该合成到最终图像的哪些区域
        # orginal_img: 原始图像，用于与重建后的图像计算损失
        # imgs: 用于重建的图像列表（在这个场景中未使用）
        # encoder_feats: 编码器提取的特征列表，与imgs一一对应
        # low_level_feats: 低级别特征列表，与imgs一一对应

        # 利用ReconstructionDecoderV2为每个图像重建生成图像
        # 初始化一个空列表来存储重建的图像
        recons_imgs = []

        # 逐个处理图像
        for img, encoder_feat, low_level_feat in zip(imgs, encoder_feats, low_level_feats):
            recons_img = self.realRecon(img, encoder_feat, low_level_feat)
            recons_imgs.append(recons_img)
            # 可选：在每次迭代后清除未使用的缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # 确定设备类型（GPU或CPU），用于数据迁移
        device = torch.device("cuda:0" if self.cfg.SYSTEM.USE_GPU else "cpu")

        # 将均值和标准差张量移动到相应的设备上
        self.mean_tensor = self.mean_tensor.cuda(device)
        self.std_tensor = self.std_tensor.cuda(device)
        # 多张mask之后的图重构的图合成一张图
        final_recon_img = sum(
            map(lambda x, y: x * (torch.tensor(1 - y, requires_grad=False).to(device)), recons_imgs, Ms))

        # 计算最终重建图像与原始图像之间的损失
        recons_loss = self.recon_loss(final_recon_img,
                                      torch.clamp(orginal_img * self.std_tensor + self.mean_tensor, 0, 1))[:, None, ...]

        # return [final_recon_img, recons_loss, bl]
        return [final_recon_img, recons_loss]


def conv_block(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )

# modified code from https://github.com/Po-Hsun-Su/pytorch-ssim
# 定义用于计算SSIM损失的类，SSIM是一种衡量图像相似性的指标，常用于图像重建质量评估
class SSIMLoss(torch.nn.Module):
    def __init__(self, window_size, absval):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.channel = 3
        self.absval = absval
        self.window = self.create_window(window_size, self.channel).float()

    def gaussian(self, window_size, sigma):
        gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
        return gauss/gauss.sum()

    def create_window(self, window_size, channel):
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
        return window

    def forward(self, recons, input):
        (_, channel, _, _) = input.size()
        if channel == self.channel and self.window.data.type() == input.data.type():
            window = self.window
        else:
            window = self.create_window(self.window_size, channel)
            if input.is_cuda:
                window = window.cuda(input.get_device())
            window = window.type_as(input)
            self.window = window
            self.channel = channel

        mu1 = F.conv2d(input, self.window, padding = self.window_size//2, groups = self.channel)
        mu2 = F.conv2d(recons, self.window, padding = self.window_size//2, groups = self.channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1*mu2

        sigma1_sq = F.conv2d(input*input, self.window, padding = self.window_size//2, groups = self.channel) - mu1_sq
        sigma2_sq = F.conv2d(recons*recons, self.window, padding = self.window_size//2, groups = self.channel) - mu2_sq
        sigma12 = F.conv2d(input*recons, self.window, padding = self.window_size//2, groups = self.channel) - mu1_mu2

        C1 = 0.01**2
        C2 = 0.03**2

        ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
        if self.absval:
            return 1-torch.clamp(torch.abs(ssim_map), 0, 1).mean(1)
        else:
            return 1-torch.clamp(ssim_map, 0, 1).mean(1)


