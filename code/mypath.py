from config import cfg
class Path(object):
    @staticmethod
    def dataset_root_dir(dataset):
        if dataset == 'cityscapes' or dataset =="cityscapes_2class":
            return str(cfg.DATASET.PATH.CITYSCAPES)     # folder that contains leftImg8bit/
        if dataset == 'LaF':
            return str(cfg.DATASET.PATH.LAF)     # folder that contains leftImg8bit/
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError
