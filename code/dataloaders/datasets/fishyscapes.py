# import bdlb
# import torch
#
# fs = bdlb.load(benchmark="fishyscapes")
# # automatically downloads the dataset
# data = fs.get_dataset('LostAndFound')
#
# # test your method with the benchmark metrics
# def estimator(image):
#     """Assigns a random uncertainty per pixel."""
#     uncertainty = torch.random(image.shape[:-1])
#     return uncertainty
#
# metrics = fs.evaluate(estimator, data.take(2))
# print('My method achieved {:.2f}% AP'.format(100 * metrics['AP']))


import torch
import os
import cv2
from torch.utils import data
from torchvision.transforms.functional import resize


def round_to_nearest_multiple(x, p):
    return int(((x - 1) // p + 1) * p)


def read_image(path):

    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

    return img


class FishyscapesLAF(data.Dataset):
    """
    The Dataset folder is assumed to follow the following structure. In the given root folder, there must be two
    sub-folders:
    - fishyscapes_lostandfound: contains the mask labels.
    - laf_images: contains the images taken from the Lost & Found Dataset
    """

    def __init__(self, root, transforms):
        super().__init__()

        self.root = root
        self.transforms = transforms

        self.images = []
        self.labels = []

        labels_path = os.path.join(
            self.root, 'fishyscapes_lostandfound')
        label_files = os.listdir(labels_path)
        label_files.sort()
        for lbl in label_files:

            self.labels.extend([os.path.join(labels_path, lbl)])
            img_name = lbl[5:-10] + 'leftImg8bit.png'
            self.images.extend(
                [os.path.join(self.root, 'laf_images', img_name)])

        self.num_samples = len(self.images)

    def __getitem__(self, index):

        image = read_image(self.images[index])
        label = read_image(self.labels[index])

        label = label[:, :, 0]

        aug = self.transforms(image=image, mask=label)
        image = aug['image']
        label = aug['mask']

        return image, label.type(torch.LongTensor)

    def __len__(self):
        return self.num_samples
