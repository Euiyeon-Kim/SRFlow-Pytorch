import os
from glob import glob

import torch
import numpy as np
from PIL import Image
import torch.utils.data as data


def random_crop(hr, size_hr):
    h, w, _ = hr.shape
    start_h = np.random.randint(low=0, high=(h - size_hr) + 1) if h > size_hr else 0
    start_w = np.random.randint(low=0, high=(w - size_hr) + 1) if w > size_hr else 0
    hr_patch = hr[start_h:start_h+size_hr, start_w:start_w+size_hr, :]
    return hr_patch


class CelebADataset(data.Dataset):
    def __init__(self, config):
        super(CelebADataset, self).__init__()
        self.config = config
        self.crop_size = config.patch_size[0]
        self.scale = config.scale
        self.lr_size = self.crop_size // self.scale

        self.gt_paths = sorted(glob(f'{config.dataset.data_root_dir}/*'))

    def __len__(self):
        return len(self.gt_paths)

    def __getitem__(self, item):
        gt = np.asarray(Image.open(self.gt_paths[item]).convert('RGB'))

        # Center Crop
        if self.config.dataset.augment.crop_center:
            hr = random_crop(gt, self.crop_size)
        # Random Crop
        else:
            hr = random_crop(gt, self.crop_size)

        # Random Flip
        if self.config.dataset.augment.use_flip:
            hr = np.flip(hr, axis=np.random.choice([True, False])).copy() if np.random.choice([True, False]) else hr

        # Random Rotate
        if self.config.dataset.augment.use_rot:
            hr = np.rot90(hr, np.random.choice([1, 2, 3])).copy() if np.random.choice([True, False]) else hr

        lr = np.asarray(Image.fromarray(hr).resize((self.lr_size, self.lr_size), Image.BICUBIC))

        lr = np.moveaxis(lr, -1, 0) / 255.0
        hr = np.moveaxis(hr, -1, 0) / 255.0

        return {'LQ': lr, 'GT': hr}


def get_train_dataloader(config):
    dataset = CelebADataset(config)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.dataset.batch_size,
                                             shuffle=config.dataset.augment.use_shuffle,
                                             num_workers=config.dataset.num_workers, drop_last=True)
    return dataloader


if __name__ == '__main__':
    from config import config as _config
    _config.dataset.data_root_dir = '../datasets/CelebA'
    dataset = CelebADataset(_config)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True, num_workers=2, drop_last=True)
    for _, train_data in enumerate(dataloader):
        print(train_data['LQ'].shape, train_data['GT'].shape)