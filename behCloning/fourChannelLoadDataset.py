#!/usr/bin/python3

import numpy as np
import os
from torch.utils.data import Dataset
from PIL import Image
import sys

IMAGE_DIR = "/recorded_images"
IMAGE_DIR2 = "/arranged_images"
IMAGE_DIR3 = "/balanced_images"


class CustomDataSet(Dataset):
    def __init__(self, main_dir, transform):
        # super().__init__()
        self.main_dir = main_dir
        self.img_dir = main_dir + IMAGE_DIR
        self.transform = transform
        self.actions = np.load(self.main_dir + '/actionsData.npy')
        self.img_names = np.load(self.main_dir + '/obsData.npy')

        # If there is an error/exception with some images, use following:
        idx = 0
        img_loc = os.path.join(self.main_dir, self.img_names[idx])
        self.dummy_image = Image.open(img_loc)
        self.dummy_label = self.actions[idx][0]
        # print(type(self.dummy_label))

    def __len__(self):
        return len(self.img_names)

    # Return image converted to tensor and the label, given the index in the dataset
    def __getitem__(self, idx):
        img_loc = os.path.join(self.main_dir, self.img_names[idx])
        image = self.dummy_image
        label = self.dummy_label
        try:
            image2 = Image.open(img_loc)  # If the file cannot be open, throw and handle exception
        except:
            print(f"Exception for {img_loc}: {sys.exc_info()[0]}")
            return self.transform(image), label

        tensor_image = self.transform(image2)
        label = self.actions[idx][0]
        return tensor_image, label


class CustomDataSet2(Dataset):
    def __init__(self, main_dir, transform):
        # super().__init__()
        self.main_dir = main_dir
        self.img_dir = main_dir + IMAGE_DIR3
        self.classes = [d.name for d in os.scandir(self.img_dir) if d.is_dir()]
        self.classes.sort()
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        # classes, class_to_idx
        self.transform = transform

        self.samples = []
        for target_class in sorted(self.class_to_idx.keys()):
            class_index = self.class_to_idx[target_class]
            target_dir = os.path.join(self.img_dir, target_class)
            if not os.path.isdir(target_dir):
                continue
            for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
                for fname in sorted(fnames):
                    path = os.path.join(root, fname)
                    # if is_valid_file(path):
                    item = path, class_index
                    self.samples.append(item)
        if len(self.samples) == 0:
            msg = "Found 0 files in subfolders of: {}\n".format(self.img_dir)
            raise RuntimeError(msg)

        self.targets = [s[1] for s in self.samples]

    def __len__(self):
        return len(self.samples)

    # Slower but simple solution
    def __getitem__(self, idx):
        img_loc, label = self.samples[idx]
        # img_loc = os.path.join(self.main_dir, self.img_names[idx])
        image = Image.open(img_loc)
        tensor_image = self.transform(image)

        return tensor_image, label
