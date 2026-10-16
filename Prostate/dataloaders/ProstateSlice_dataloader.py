import math
from torch.utils import data
import numpy as np
from PIL import Image
from batchgenerators.utilities.file_and_folder_operations import *
from dataloaders.normalize import normalize_image, normalize_image_to_0_1_3D
import matplotlib.pyplot as plt


class ProstateSlice_dataset(data.Dataset):
    def __init__(self, root, img_list, label_list, target_size=512, batch_size=None, img_normalize=True):
        super().__init__()
        self.root = root
        self.img_list = img_list
        self.label_list = label_list
        self.len = len(img_list)
        self.target_size = (target_size, target_size)
        self.img_normalize = img_normalize
        # if batch_size is not None:
        #     iter_nums = len(self.img_list) // batch_size
        #     scale = math.ceil(100 / iter_nums)
        #     self.img_list = self.img_list * scale
        #     self.label_list = self.label_list * scale

    def __len__(self):
        return self.len

    def __getitem__(self, item):
        img_file = os.path.join(self.root, self.img_list[item])
        label_file = os.path.join(self.root, self.label_list[item])
        img = Image.open(img_file)
        label = Image.open(label_file).convert('L')

        img = img.resize(self.target_size)
        label = label.resize(self.target_size, resample=Image.NEAREST)
        img_npy = np.array(img)[np.newaxis, ...].astype(np.float32)
        if self.img_normalize:
            img_npy = normalize_image_to_0_1_3D(img_npy)
        label_npy = np.array(label)

        mask = np.zeros_like(label_npy)
        mask[label_npy < 255] = 1
        return img_npy, mask[np.newaxis], img_file

