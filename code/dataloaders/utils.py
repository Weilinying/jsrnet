import matplotlib.pyplot as plt
import numpy as np
import torch

def decode_seg_map_sequence(label_masks, dataset):
    rgb_masks = []
    # 遍历所有标签掩码
    for label_mask in label_masks:
        # 解码单个标签掩码为RGB图像
        rgb_mask = decode_segmap(label_mask, dataset)
        rgb_masks.append(rgb_mask)
    # 转换列表为张量，并调整维度以符合PyTorch期望的维度顺序
    rgb_masks = torch.from_numpy(np.array(rgb_masks).transpose([0, 3, 1, 2]))
    return rgb_masks

def decode_segmap(label_mask, dataset, plot=False):
    """Decode segmentation class labels into a color image  解码分割类别标签到彩色图像
    Args:
        label_mask (np.ndarray): an (M,N) array of integer values denoting  一个(M,N)的整数数组，表示每个空间位置的类别标签。
          the class label at each spatial location.
        plot (bool, optional): whether to show the resulting color image
          in a figure.
    Returns:
        (np.ndarray, optional): the resulting decoded color image.  结果解码的彩色图像。
    """
    """
        将分割类标签解码成彩色图像。

        Args:
            label_mask (np.ndarray): 一个(M,N)的整数数组，表示每个空间位置的类标签。
            dataset (str): 指定数据集名称，不同的数据集有不同的类别和颜色编码。
            plot (bool, optional): 是否在图中显示结果彩色图像，默认为False。

        Returns:
            np.ndarray, optional: 解码后的彩色图像。
    """
    if dataset == 'cityscapes' or dataset == "cityscapes_2class" or 'LaF':
        n_classes = 19
        label_colours = get_cityscapes_labels()
    else:
        raise NotImplementedError

    r = label_mask.copy()
    g = label_mask.copy()
    b = label_mask.copy()
    # 根据类别给每个像素指定颜色
    for ll in range(0, n_classes):
        r[label_mask == ll] = label_colours[ll, 0]
        g[label_mask == ll] = label_colours[ll, 1]
        b[label_mask == ll] = label_colours[ll, 2]
    rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
    # 归一化
    rgb[:, :, 0] = r / 255.0
    rgb[:, :, 1] = g / 255.0
    rgb[:, :, 2] = b / 255.0
    # 根据plot参数的值来决定是直接显示RGB图像还是返回RGB图像的数据
    if plot:
        plt.imshow(rgb)
        plt.show()
    else:
        return rgb

def encode_segmap(mask):
    """Encode segmentation label images as pascal classes  将分割标签图像编码为类别
    Args:
        mask (np.ndarray): raw segmentation label image of dimension  原始分割标签图像，维度为(M, N, 3)，其中Pascal类别以颜色编码。
          (M, N, 3), in which the Pascal classes are encoded as colours.
    Returns:
        (np.ndarray): class map with dimensions (M,N), where the value at  类别图，维度为(M,N)，给定位置的值为表示类别索引的整数。
        a given location is the integer denoting the class index.
    """
    mask = mask.astype(int)
    label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int16)
    # 将颜色映射到类别索引
    for ii, label in enumerate(get_pascal_labels()):
        label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = ii
    label_mask = label_mask.astype(int)
    return label_mask

def get_cityscapes_labels():
    # 返回Cityscapes数据集的标签颜色映射
    return np.array([
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [0, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32]])

