from dataloaders.datasets import cityscapes, cityscapes_2class, lostandfound, obstacles_track
from torch.utils.data import DataLoader
from mypath import Path

# 定义一个函数来根据配置创建数据加载器
def make_data_loader(cfg, **kwargs):
    # 根据配置选择训练集，并实例化相应的数据集对象
    if cfg.DATASET.TRAIN == 'cityscapes':
        train_set = cityscapes.CityscapesSegmentation(cfg, root=Path.dataset_root_dir(cfg.DATASET.TRAIN), split='train')
    elif cfg.DATASET.TRAIN == 'cityscapes_2class':
        train_set = cityscapes_2class.CityscapesSegmentation_2Class(cfg, root=Path.dataset_root_dir(cfg.DATASET.TRAIN), split='train')
    elif cfg.DATASET.TRAIN == 'LaF':
        train_set=lostandfound.LostAndFound(cfg, root=Path.dataset_root_dir(cfg.DATASET.TRAIN), split='train')
    else:
        raise NotImplementedError

    # 类似地选择验证集，并实例化数据集对象
    if cfg.DATASET.VAL == 'cityscapes':
        val_set = cityscapes.CityscapesSegmentation(cfg, root=Path.dataset_root_dir(cfg.DATASET.VAL), split='val')
    elif cfg.DATASET.VAL == 'cityscapes_2class':
        val_set = cityscapes_2class.CityscapesSegmentation_2Class(cfg, root=Path.dataset_root_dir(cfg.DATASET.VAL), split='val')
    elif cfg.DATASET.VAL == 'LaF':
        val_set = lostandfound.LostAndFound(cfg, root=Path.dataset_root_dir(cfg.DATASET.TEST), split='val')
    elif cfg.DATASET.VAL == 'OT':
        val_set = obstacles_track.RoadObstacle21(cfg, root=Path.dataset_root_dir(cfg.DATASET.VAL), split='val')
    else:
        raise NotImplementedError

    # 选择测试集，并实例化数据集对象
    if cfg.DATASET.TEST == 'cityscapes':
        test_set = cityscapes.CityscapesSegmentation(cfg, root=Path.dataset_root_dir(cfg.DATASET.TEST), split='test')
    elif cfg.DATASET.TEST == 'cityscapes_2class':
        test_set = cityscapes_2class.CityscapesSegmentation_2Class(cfg, root=Path.dataset_root_dir(cfg.DATASET.TEST), split='test')
    elif cfg.DATASET.TEST == 'LaF':
        test_set = lostandfound.LostAndFound(cfg, root=Path.dataset_root_dir(cfg.DATASET.TEST), split='test')
    elif cfg.DATASET.TEST == 'OT':
        test_set = obstacles_track.RoadObstacle21(cfg, root=Path.dataset_root_dir(cfg.DATASET.TEST), split='val')
    else:
        raise NotImplementedError

    # 使用DataLoader工具创建数据加载器，设置批次大小、是否打乱顺序等参数
    train_loader = DataLoader(train_set, batch_size=cfg.INPUT.BATCH_SIZE_TRAIN, shuffle=True, **kwargs)
    val_loader = DataLoader(val_set, batch_size=cfg.INPUT.BATCH_SIZE_TEST, shuffle=False, **kwargs)
    test_loader = DataLoader(test_set, batch_size=cfg.INPUT.BATCH_SIZE_TEST, shuffle=False, **kwargs)
    # 返回训练、验证、测试的数据加载器以及训练数据集中的类别数量
    return train_loader, val_loader, test_loader, train_set.NUM_CLASSES
