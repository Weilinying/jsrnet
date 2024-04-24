import os
import cv2
import torch
import numpy as np
import webp
from torchvision import transforms

from config import cfg
from dataloaders import custom_transforms as tr
from torch.utils import data


class RoadAnomaly21(data.Dataset):
    """
    The given dataset_root entry in hparams is expected to be the root folder that contains the Anomaly Track data of
    the SegmentMeIfYouCan benchmark. The contents of that folder are 'images/', 'label_masks/', and 'LICENSE.txt'.
    Not all images are expected to have ground truth labels because the test set is held-out. However for consistency
    the dataset __getitem__ will return two tensors, the image and a dummy label (if the original one does not exist).

    The dataset has 10 extra validation samples with ground truth labels. Therefore in this dataset we define 3 modes of
    the data:
    - test: only loads the test samples without existing labels
    - val: only loads the validation samples that have ground truth labels. They are distinguished by the filenames which contain
        the string 'validation'.
    - all: loads everything including validation and testing samples.

    """

    def __init__(self, cfg, root, split):
        super().__init__()

        self.root = root
        self.split = split
        self.base_size = cfg.INPUT.BASE_SIZE  # 基础尺寸
        self.crop_size = cfg.INPUT.CROP_SIZE  # 裁剪尺寸
        # 图像归一化参数
        self.img_mean = cfg.INPUT.NORM_MEAN
        self.img_std = cfg.INPUT.NORM_STD

        # 忽略的标签索引
        self.ignore_index = cfg.LOSS.IGNORE_LABEL

        images_root = os.path.join(self.root, 'images')
        labels_root = os.path.join(self.root, 'labels_masks')
        self.images = [os.path.join(images_root, f) for f in os.listdir(images_root) if 'validation' in f]

        # 根据数据模式，筛选出相应的图像和标签文件
        if self.split == 'test':
            # 仅加载非验证样本的图像
            self.images = [os.path.join(images_root, img_path) for img_path in self.images if
                           not ('validation' in img_path)]
            self.labels = [''] * len(self.images)
        elif self.split == 'val':
            # 仅加载验证样本的图像和标签
            self.labels = [os.path.join(labels_root, os.path.splitext(f)[0] + '_labels_semantic.png')
                           for f in os.listdir(images_root) if 'validation' in f]
            self.images = [os.path.join(images_root, img_path) for img_path in self.images if 'validation' in img_path]
        elif self.split == 'all':
            # 加载所有样本的图像和标签
            self.labels = []
            for img_path in self.images:
                if 'validation' in img_path:
                    label_path = os.path.join(labels_root, os.path.splitext(img_path)[0] + '_labels_semantic.png')
                    self.labels.append(label_path)  # 添加到标签列表
                else:
                    self.labels.extend([''])

            self.images = [os.path.join(self.root, img_path) for img_path in self.images]
        else:
            raise Exception(f'Undefined Dataset Mode for RoadObstacle21: {cfg.dataset_mode}')

        # 数据集样本总数
        self.num_samples = len(self.images)

        # 图像转换 - 测试和验证
        self.image_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.img_mean, std=self.img_std)
        ])

        # 标签转换 - 只需调整大小
        self.label_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        # 根据索引获取图像和标签
        image = self.read_webp(self.images[index])

        if 'validation' in self.images[index]:
            label = self.read_image(self.labels[index])
        else:
            label = np.zeros_like(image)

        label = label[:, :, 0]

        # 应用图像和标签的转换
        # 应用图像的转换
        if self.image_transform is not None:
            image = self.image_transform(image)  # 确保 transform 适用于图像

            # 应用标签的转换，这里只做简单的转换，确保尺寸一致
        if self.label_transform is not None:
            label = self.label_transform(label)  # label_transform 应只包含尺寸调整和转换为张量

        return {'image': image, 'label': label}

    def __len__(self):
        return self.num_samples

    @staticmethod
    def read_image(path):

        img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

        return img
