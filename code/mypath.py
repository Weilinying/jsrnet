from config import cfg
class Path(object):
    @staticmethod
    def dataset_root_dir(dataset):
        if dataset == 'cityscapes' or dataset =="cityscapes_2class":
            return str(cfg.DATASET.PATH.CITYSCAPES)     # folder that contains leftImg8bit/
        if dataset == 'LaF':
            return str(cfg.DATASET.PATH.LAF)     # folder that contains leftImg8bit/
        if dataset == 'OT':
            return str(cfg.DATASET.PATH.OBSTACLE_TRACK)
        if dataset == 'RA21':
            return str(cfg.DATASET.PATH.ROADANOMALY21)
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError
