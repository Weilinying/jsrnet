import os
import cv2
import torch
import numpy as np
import webp
from torchvision.transforms import InterpolationMode

from config import cfg
from torch.utils import data
from torchvision import transforms
from dataloaders import custom_transforms as tr
from mypath import Path
class RoadObstacle21(data.Dataset):
    """
    The given dataset_root entry in hparams is expected to be the root folder that contains the Obstacle Track data of
    the SegmentMeIfYouCan benchmark. The contents of that folder are 'images/', 'label_masks/', and 'image-sources.txt'.
    Not all images are expected to have ground truth labels because the test set is held-out. However for consistency
    the dataset __getitem__ will return two tensors, the image and a dummy label (if the original one does not exist).

    The dataset has 30 extra validation samples with ground truth labels. Therefore in this dataset we define 3 modes of
    the data:
    - test: only loads the test samples without existing labels
    - val: only loads the validation samples that have ground truth labels. They are distinguished by the filenames which contain
        the string 'validation'.
    - all: loads everything including validation and testing samples.

    """

    def __init__(self, root, split, transforms):
        super().__init__()

        self.root = root
        self.split = split
        self.transforms = transforms

        images_root = os.path.join(self.root, 'images')
        labels_root = os.path.join(self.root, 'labels_masks')
        self.images = os.listdir(images_root)

        if self.split == 'test':

            self.images = [os.path.join(images_root, img_path) for img_path in self.images if
                           not ('validation' in img_path)]
            self.labels = [''] * len(self.images)
        elif self.split == 'val':

            self.labels = [os.path.join(labels_root, img_path[:-5] + '_labels_semantic.png') for img_path in self.images
                           if 'validation' in img_path]
            self.images = [os.path.join(images_root, img_path) for img_path in self.images if 'validation' in img_path]
        elif self.split == 'all':

            self.labels = []
            for img_path in self.images:
                if 'validation' in img_path:
                    self.labels.extend([os.path.join(labels_root, img_path[:-5] + '_labels_semantic.png')])
                else:
                    self.labels.extend([''])

            self.images = [os.path.join(images_root, img_path) for img_path in self.images]
        else:
            raise Exception(f'Undefined Dataset Mode for RoadObstacle21: {self.split}')

        self.num_samples = len(self.images)

    def __getitem__(self, index):

        image = self.read_webp(self.images[index])

        if 'validation' in self.images[index]:
            label = self.read_image(self.labels[index])
        else:
            label = np.zeros_like(image)

        label = label[:, :, 0]

        if self.transforms is not None:
            aug = self.transforms(image=image, mask=label)
            image = aug['image']
            label = aug['mask']

        sample = {'image': image, 'label': label.type(torch.LongTensor)}

        return sample

    def __len__(self):
        return self.num_samples

    @staticmethod
    def read_image(path):

        img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

        return img

    @staticmethod
    def read_webp(path):

        img = webp.load_image(path, 'RGB')
        img = np.array(img)

        return img


if __name__ == '__main__':
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    # Define a simple transformation
    transform = A.Compose([
        ToTensorV2()
    ])

    # Create an instance of the dataset
    dataset = RoadObstacle21(root=r'D:\SoftwareEngineer\graduation_project\datasets\dataset_ObstacleTrack',
                             split='val',
                             transforms=transform)

    # Access the first item
    sample = dataset[0]  # Get the sample which is a dictionary
    image = sample['image']
    label = sample['label']

    # Print the shapes of the image and label
    print(f'Image shape: {image.shape}')
    print(f'Label shape: {label.shape}')