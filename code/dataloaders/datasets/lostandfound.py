import os
import numpy as np
import scipy.misc as m
from PIL import Image
from torch.utils import data
from torchvision import transforms
from dataloaders import custom_transforms as tr
from mypath import Path
from config import cfg

class LostAndFound(data.Dataset):
    NUM_CLASSES = 2 

    def __init__(self, cfg, root, split):

        self.root = root
        self.split = split
        self.files = {}
        self.base_size = cfg.INPUT.BASE_SIZE  # 基础尺寸
        self.crop_size = cfg.INPUT.CROP_SIZE  # 裁剪尺寸
        # 图像归一化参数
        self.img_mean = cfg.INPUT.NORM_MEAN 
        self.img_std = cfg.INPUT.NORM_STD

        # 图像和标签的基础路径
        if self.split == "val":
            self.images_base = os.path.join(self.root, 'leftImg8bit', "train")
            self.annotations_base = os.path.join(self.root, 'gtCoarse', "train")
        else:
            self.images_base = os.path.join(self.root, 'leftImg8bit', self.split)
            self.annotations_base = os.path.join(self.root, 'gtCoarse', self.split)

        # 加载图像文件路径
        self.files[split] = self.recursive_glob(rootdir=self.images_base, suffix='.png')

        # 定义无效类和有效类别
        # for LaF,0 represent Random non-hazards,255  represent,so void_classes include 0 and 255
        self.void_classes = [255, 0]
        self.valid_classes = [1]

        # 忽略的标签索引
        self.ignore_index = cfg.LOSS.IGNORE_LABEL
        # 有效类别映射
        self.class_map = dict(zip([0, 7], range(self.NUM_CLASSES)))

        # 训练时的图像转换
        self.transform_tr = transforms.Compose([
            tr.RandomSizeAndCrop(self.crop_size, False, pre_size=None, scale_min=cfg.AUG.SCALE_MIN,
                                           scale_max=cfg.AUG.SCALE_MAX, ignore_index=self.ignore_index),
            tr.RandomHorizontalFlip(),
            tr.RandomRoadObject(rcp=cfg.AUG.RANDOM_CROP_PROB),
            tr.ColorJitter(brightness=cfg.AUG.COLOR_AUG, contrast=cfg.AUG.COLOR_AUG, 
                           saturation=cfg.AUG.COLOR_AUG, hue=cfg.AUG.COLOR_AUG),
            tr.RandomGaussianBlur(),
            tr.Normalize(mean=self.img_mean, std=self.img_std),
            tr.ToTensor()])
        # 验证和测试时的图像转换
        self.transform_ts = transforms.Compose([
            tr.Resize(self.crop_size),
            tr.Normalize(mean=self.img_mean, std=self.img_std),
            tr.ToTensor()])

        # 检查是否有可用图像
        if not self.files[split]:
            raise Exception("No files for split=[%s] found in %s" % (split, self.images_base))

        # 输出找到的图像数量及其所属的数据集划分类型
        print("Found %d %s images" % (len(self.files[split]), split))

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        # 加载图像和标签
        # 从文件列表中获取当前索引下的图像路径，并去除尾部空白字符。
        img_path = self.files[self.split][index].rstrip()
        # 构造标签文件的路径。这里使用图像文件路径来定位同名的标签文件，但替换文件名的尾部以指向对应的标签文件。
        lbl_path = os.path.join(self.annotations_base,
                                img_path.split(os.sep)[-2],
                                os.path.basename(img_path)[:-15] + 'gtCoarse_labelTrainIds.png')

        # 使用PIL库打开图像文件，并确保图像是RGB格式。
        _img = Image.open(img_path).convert('RGB')
        _tmp = np.array(Image.open(lbl_path), dtype=np.uint8)
        _tmp = self.encode_segmap(_tmp)  # 调用encode_segmap方法处理标签图，将原始标签映射到训练过程中使用的标签索引。
        _target = Image.fromarray(_tmp.astype(np.uint8))  # 将处理后的标签图转换回PIL图像格式，以便后续的图像转换处理。

        # 创建了一个字典sample，包含两个键值对：'image'键存储了变量_img（即处理后的图像），而'label'键存储了变量_target（即对应的处理后的标签图）。
        sample = {'image': _img, 'label': _target, 'randseed': index}

        # 根据数据集划分应用不同的图像转换
        if self.split == 'train':
            sample = self.transform_tr(sample)
        elif self.split == 'val':
            sample = self.transform_ts(sample)
        elif self.split == 'test':
            sample = self.transform_ts(sample)

        # 保存图像名称，去除扩展名
        sample["image_name"] = os.path.basename(img_path)[:-4]
        return sample

    # 编码分割图，将有效类别映射到连续的索引，无效类别设置为忽略索引
    def encode_segmap(self, mask):
        # Put all void classes to ignore
        mask_tmp = np.zeros_like(mask)
        for _voidc in self.void_classes:
            mask_tmp[mask == _voidc] = self.ignore_index
        for _validc in self.valid_classes:
            mask_tmp[mask == _validc] = 1 
        return mask_tmp

    # 递归搜索特定后缀的文件
    def recursive_glob(self, rootdir='.', suffix=''):
        """Performs recursive glob with given suffix and rootdir
            :param rootdir is the root directory
            :param suffix is the suffix to be searched
        """
        return [os.path.join(looproot, filename)
                for looproot, _, filenames in os.walk(rootdir)
                for filename in filenames if filename.endswith(suffix)]


if __name__ == '__main__':
    from dataloaders.utils import decode_segmap
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    from yacs.config import CfgNode as CN

    # 实例化数据集对象
    cityscapes_train = LostAndFound(cfg, Path.dataset_root_dir('LaF'), split='train')

    # 创建DataLoader
    dataloader = DataLoader(cityscapes_train, batch_size=8, shuffle=True, num_workers=4)

    # 遍历数据集并显示图像及其分割图
    for ii, sample in enumerate(dataloader):
        for jj in range(sample["image"].size()[0]):
            img = sample['image'].numpy()  # 将图像从Tensor转换为numpy数组
            gt = sample['label'].numpy()  # 将标签从Tensor转换为numpy数组
            tmp = np.array(gt[jj]).astype(np.uint8)  # 将当前标签转换为uint8类型
            segmap = decode_segmap(tmp, dataset='cityscapes')  # 解码标签图为可视化的分割图
            img_tmp = np.transpose(img[jj], axes=[1, 2, 0])  # 重新排列轴，以便图像具有正确的维度
            img_tmp *= (0.229, 0.224, 0.225)  # 反归一化图像（乘以标准差）
            img_tmp += (0.485, 0.456, 0.406)  # 反归一化图像（加上均值）
            img_tmp *= 255.0  # 将图像的像素值从[0, 1]转换为[0, 255]
            img_tmp = img_tmp.astype(np.uint8)  # 确保图像的类型为uint8
            plt.figure()  # 创建一个新的图形
            plt.title('display')  # 设置图形的标题
            plt.subplot(211)  # 在图形的上半部分绘制图像
            plt.imshow(img_tmp)  # 显示图像
            plt.subplot(212)  # 在图形的下半部分绘制分割图
            plt.imshow(segmap)  # 显示分割图

        if ii == 1:  # 仅显示两个批次的数据
            break
    # 显示所有图形
    plt.show(block=True)

